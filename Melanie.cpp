// puzzle.cpp — moteur de recherche "Bitcoin puzzle" : UN SEUL fichier, deux moteurs.
//
// Ce fichier detecte TOUT SEUL, a la compilation, s'il est compile par un compilateur C++
// normal (g++, clang++...) ou par le compilateur CUDA de NVIDIA (nvcc) :
//   - compilateur normal -> moteur CPU pur (SIMD, fenetre +-i, aucune dependance externe).
//   - nvcc                -> moteur GPU (CUDA), utilise TOUTES les cartes visibles automatiquement
//                            (ex : les 2 T4 d'un notebook Kaggle), sans rien configurer de plus.
// La detection se fait via la macro standard __CUDACC__, que seul nvcc definit quand il traite
// un fichier comme du code CUDA.
//
// Compilation :
//   CPU (n'importe quelle machine)        : g++ -O3 -march=native -std=c++17 -o puzzle puzzle.cpp -pthread
//   GPU (Kaggle, notebook "GPU T4 x2")     : nvcc -x cu -O3 -std=c++17 -arch=sm_75 -o puzzle puzzle.cpp
//
// IMPORTANT pour la voie GPU : comme ce fichier porte l'extension .cpp (pas .cu), nvcc ne le
// traite PAS comme du CUDA par defaut — il faut lui dire explicitement avec "-x cu" (sinon il
// delegue tout au compilateur hote normal, ce qui revient a compiler le moteur CPU avec nvcc,
// sans erreur mais sans le GPU non plus). C'est "-x cu" qui fait que __CUDACC__ devient defini
// et qui active donc le moteur GPU ci-dessous. T4 = architecture Turing = compute capability 7.5.
//
// Usage (identique dans les deux moteurs) :
//   ./puzzle -pP                      (puzzle numero P, 1-80)
//   ./puzzle -pApB                    (tous les puzzles de A a B, cherches EN MEME TEMPS)
//   ./puzzle <adresse> <bits 1-80>    (plage [2^(bits-1), 2^bits-1])
//   ./puzzle <adresse> <debut:fin>    (plage personnalisee, en hexadecimal)
//   -vN / -vvN / -vvvN / -vvvvN       (mode voisinage : prefixe / milieu / suffixe / les trois, N bits)
//   -s T                              (rafraichit l'affichage toutes les T secondes, defaut 30)
// Options propres au moteur CPU : -t N (nombre de coeurs), --seq (ordre sequentiel), --chunk N.
// Options propres au moteur GPU : -g N (taille de lot = 2^N cles par lancement de kernel, defaut 23).
// Une option propre a l'AUTRE moteur est simplement ignoree (avec un avertissement), pas une erreur.
//
// HISTORIQUE ET CONCEPTION DU MOTEUR CPU (inchange depuis puzzle_pur.cpp) :
//   - U256 (4 limbes fixes) + reduction de Solinas dediee a p = 2^256 - 2^32 - 977.
//   - Carre dedie (fe_sqr), inversion modulaire par chaine d'addition fixe (fe_inv).
//   - Marche sur la courbe par fenetre centree +-i (scan_batch/WinTable) : une inversion
//     partagee pour 2*HALF=1024 cles a la fois (astuce a la Montgomery).
//   - hash160 par paquets de 8 cles en SIMD (hash160_vec).
//   - Mode voisinage (-v) : cellules visitees dans un ordre aleatoire SANS DOUBLON (permutation
//     de Feistel), un thread independant par tentative.
//   - Recherche multi-cibles (-pApB) : filtre binaire (bloom) + liste de hash160, un seul passage
//     sur la courbe sert a comparer a toutes les cibles a la fois.
//
// CONCEPTION DU MOTEUR GPU (ajoute pour Kaggle, 2x T4) :
//   - Arithmetique de corps et courbe RE-IMPLEMENTEES cote device (prefixe d_), sans unsigned
//     __int128 (non garanti en code __device__) : chaque multiplication 64x64->128 bits passe par
//     l'intrinseque CUDA __umul64hi. hash160 device = portage direct des fonctions scalaires du
//     moteur CPU (pas la version SIMD, inutile : chaque thread CUDA est deja une "voie").
//   - Chaque thread GPU traite sa cle de bout en bout, inversion modulaire comprise, de facon
//     independante (plus simple et plus sur a porter sans GPU disponible pour tester que la
//     technique d'inversion par lot du CPU — une optimisation possible pour plus tard).
//   - Mode voisinage GPU : un lancement de kernel regroupe PLUSIEURS cellules de voisinage a la
//     fois (scan_kernel_cells), pour saturer le GPU meme quand une cellule est minuscule.
//   - Un auto-test tourne sur chaque carte au demarrage (gpu_self_test / gpu_neighbor_self_test) :
//     il lance le vrai kernel sur des cles connues et verifie le resultat avant tout vrai balayage.
//
// Les deux moteurs partagent tout le reste a l'identique : arithmetique, decodage d'adresse,
// format de sortie (God.txt), table des puzzles, arret propre par Ctrl-C.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

typedef uint64_t u64;
typedef uint32_t u32;
typedef unsigned __int128 u128; // cote HOTE uniquement (jamais dans une fonction __device__)

#ifdef __CUDACC__
#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        std::cerr << "Erreur CUDA (" << __FILE__ << ":" << __LINE__ << ") : " \
                  << cudaGetErrorString(_e) << "\n"; \
        std::exit(1); \
    } \
} while (0)
#endif

struct U256 { u64 d[4]; };
static const U256 ZERO = {{0,0,0,0}};
static const U256 ONE  = {{1,0,0,0}};
static const u64  KC = 0x1000003D1ULL; // 2^256 mod p (p = 2^256 - 2^32 - 977)

static inline bool is_zero(const U256 &a) { return (a.d[0]|a.d[1]|a.d[2]|a.d[3]) == 0; }
static inline bool eq(const U256 &a, const U256 &b) {
    return ((a.d[0]^b.d[0])|(a.d[1]^b.d[1])|(a.d[2]^b.d[2])|(a.d[3]^b.d[3])) == 0;
}
static inline u64 add_raw(U256 &r, const U256 &a, const U256 &b) {
    u128 acc = 0; u64 c = 0;
    for (int i = 0; i < 4; i++) { acc = (u128)a.d[i] + b.d[i] + c; r.d[i] = (u64)acc; c = (u64)(acc >> 64); }
    return c;
}
static inline u64 sub_raw(U256 &r, const U256 &a, const U256 &b) {
    u64 br = 0;
    for (int i = 0; i < 4; i++) {
        u128 t = (u128)a.d[i] - b.d[i] - br;
        r.d[i] = (u64)t; br = (u64)((t >> 64) & 1);
    }
    return br;
}
static int highest_bit(const U256 &a) {
    for (int i = 255; i >= 0; --i) if ((a.d[i >> 6] >> (i & 63)) & 1) return i;
    return -1;
}
static inline int get_bit(const U256 &a, int i) { return (int)((a.d[i >> 6] >> (i & 63)) & 1); }

static bool parse_hex(const std::string &s, U256 &out) {
    if (s.empty() || s.size() > 64) return false;
    out = ZERO;
    for (size_t i = 0; i < s.size(); i++) {
        char c = s[s.size() - 1 - i]; int v;
        if (c >= '0' && c <= '9') v = c - '0'; else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v = c - 'A' + 10; else return false;
        out.d[i / 16] |= (u64)v << ((i % 16) * 4);
    }
    return true;
}
static U256 u128_to_U256(u128 v) { U256 r; r.d[0] = (u64)v; r.d[1] = (u64)(v >> 64); r.d[2] = 0; r.d[3] = 0; return r; }

// =================== Corps fini Fp (secp256k1) ===================
static inline void fe_add(U256 &r, const U256 &a, const U256 &b) {
    U256 t; u64 c = add_raw(t, a, b);
    if (c) { U256 k = {{KC,0,0,0}}; add_raw(t, t, k); }
    // t >= p ? soustraire p — comparaison directe plutôt que de stocker p entier à part
    static const U256 P = {{0xFFFFFFFEFFFFFC2FULL, ~0ULL, ~0ULL, ~0ULL}};
    bool geq = true;
    for (int i = 3; i >= 0 && geq; --i) if (t.d[i] != P.d[i]) { geq = t.d[i] > P.d[i]; break; }
    if (geq) sub_raw(t, t, P);
    r = t;
}
static inline void fe_sub(U256 &r, const U256 &a, const U256 &b) {
    U256 t; u64 br = sub_raw(t, a, b);
    if (br) { U256 k = {{KC,0,0,0}}; sub_raw(t, t, k); }
    r = t;
}
static inline void fe_reduce(U256 &r, const u64 *t) {
    u64 r5[5]; u64 carry = 0;
    for (int i = 0; i < 4; i++) { u128 acc = (u128)t[4+i]*KC + t[i] + carry; r5[i] = (u64)acc; carry = (u64)(acc>>64); }
    r5[4] = carry;
    U256 res; u128 acc = (u128)r5[4]*KC + r5[0];
    res.d[0] = (u64)acc; carry = (u64)(acc>>64);
    for (int i = 1; i < 4; i++) { acc = (u128)r5[i]+carry; res.d[i]=(u64)acc; carry=(u64)(acc>>64); }
    if (carry) {
        acc = (u128)res.d[0]+KC; res.d[0]=(u64)acc; carry=(u64)(acc>>64);
        for (int i = 1; i < 4; i++) { acc=(u128)res.d[i]+carry; res.d[i]=(u64)acc; carry=(u64)(acc>>64); }
    }
    static const U256 P = {{0xFFFFFFFEFFFFFC2FULL, ~0ULL, ~0ULL, ~0ULL}};
    bool geq = true;
    for (int i = 3; i >= 0; --i) if (res.d[i] != P.d[i]) { geq = res.d[i] > P.d[i]; break; }
    if (geq) sub_raw(res, res, P);
    r = res;
}
static inline void fe_mul(U256 &r, const U256 &a, const U256 &b) {
    u64 t[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++) { u128 acc=(u128)a.d[i]*b.d[j]+t[i+j]+carry; t[i+j]=(u64)acc; carry=(u64)(acc>>64); }
        t[i+4] = carry;
    }
    fe_reduce(r, t);
}
static inline void fe_sqr(U256 &r, const U256 &a) { // carré dédié : 10 produits au lieu de 16
    u64 t[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 3; i++) {
        u64 carry = 0;
        for (int j = i+1; j < 4; j++) { u128 acc=(u128)a.d[i]*a.d[j]+t[i+j]+carry; t[i+j]=(u64)acc; carry=(u64)(acc>>64); }
        t[i+4] = carry;
    }
    u64 c = 0;
    for (int k = 0; k < 8; k++) { u64 v=t[k]; t[k]=(v<<1)|c; c=v>>63; }
    u64 carry = 0;
    for (int i = 0; i < 4; i++) {
        u128 acc=(u128)a.d[i]*a.d[i]+t[2*i]+carry; t[2*i]=(u64)acc; carry=(u64)(acc>>64);
        acc=(u128)t[2*i+1]+carry; t[2*i+1]=(u64)acc; carry=(u64)(acc>>64);
    }
    fe_reduce(r, t);
}
static inline void fe_sqr_n(U256 &r, const U256 &a, int n) { U256 t=a; for(int i=0;i<n;i++) fe_sqr(t,t); r=t; }
// Inversion par chaîne d'addition fixe (équivalent à a^(p-2) mod p, 255 carrés + 14 produits)
static U256 fe_inv(const U256 &a) {
    U256 x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t;
    fe_sqr(x2,a); fe_mul(x2,x2,a);
    fe_sqr(x3,x2); fe_mul(x3,x3,a);
    fe_sqr_n(x6,x3,3); fe_mul(x6,x6,x3);
    fe_sqr_n(x9,x6,3); fe_mul(x9,x9,x3);
    fe_sqr_n(x11,x9,2); fe_mul(x11,x11,x2);
    fe_sqr_n(x22,x11,11); fe_mul(x22,x22,x11);
    fe_sqr_n(x44,x22,22); fe_mul(x44,x44,x22);
    fe_sqr_n(x88,x44,44); fe_mul(x88,x88,x44);
    fe_sqr_n(x176,x88,88); fe_mul(x176,x176,x88);
    fe_sqr_n(x220,x176,44); fe_mul(x220,x220,x44);
    fe_sqr_n(x223,x220,3); fe_mul(x223,x223,x3);
    fe_sqr_n(t,x223,23); fe_mul(t,t,x22);
    fe_sqr_n(t,t,5); fe_mul(t,t,a);
    fe_sqr_n(t,t,3); fe_mul(t,t,x2);
    fe_sqr_n(t,t,2); fe_mul(t,t,a);
    return t;
}

// =================== Courbe secp256k1 (y^2 = x^3 + 7) ===================
struct Point { U256 x, y; bool inf = false; };
struct Jac { U256 X, Y, Z; bool inf = false; };

static Point G;

static Point aff_add(const Point &P, const Point &Q) {
    if (P.inf) return Q;
    if (Q.inf) return P;
    U256 l;
    if (eq(P.x, Q.x)) {
        if (!eq(P.y, Q.y) || is_zero(P.y)) return Point{ZERO, ZERO, true};
        U256 n, d, n3; fe_sqr(n, P.x); fe_add(n3, n, n); fe_add(n3, n3, n);
        fe_add(d, P.y, P.y); fe_mul(l, n3, fe_inv(d));
    } else {
        U256 n, d; fe_sub(n, Q.y, P.y); fe_sub(d, Q.x, P.x); fe_mul(l, n, fe_inv(d));
    }
    Point R; R.inf = false;
    fe_sqr(R.x, l); fe_sub(R.x, R.x, P.x); fe_sub(R.x, R.x, Q.x);
    U256 t; fe_sub(t, P.x, R.x); fe_mul(R.y, l, t); fe_sub(R.y, R.y, P.y);
    return R;
}
static Jac jac_dbl(const Jac &P) {
    if (P.inf || is_zero(P.Y)) return Jac{ZERO, ZERO, ZERO, true};
    U256 A,B,C,D,E,F,t,t2;
    fe_sqr(A,P.X); fe_sqr(B,P.Y); fe_sqr(C,B);
    fe_add(t,P.X,B); fe_sqr(t,t); fe_sub(t,t,A); fe_sub(t,t,C); fe_add(D,t,t);
    fe_add(E,A,A); fe_add(E,E,A); fe_sqr(F,E);
    Jac R; R.inf=false;
    fe_add(t,D,D); fe_sub(R.X,F,t);
    fe_sub(t,D,R.X); fe_mul(t,E,t);
    fe_add(t2,C,C); fe_add(t2,t2,t2); fe_add(t2,t2,t2);
    fe_sub(R.Y,t,t2);
    fe_mul(t,P.Y,P.Z); fe_add(R.Z,t,t);
    return R;
}
static Jac jac_madd(const Jac &P, const Point &Q) {
    if (P.inf) { Jac R; R.X=Q.x; R.Y=Q.y; R.Z=ONE; R.inf=false; return R; }
    U256 Z1Z1,U2,S2,H,HH,I,J,r,V,t;
    fe_sqr(Z1Z1,P.Z); fe_mul(U2,Q.x,Z1Z1);
    fe_mul(S2,Q.y,P.Z); fe_mul(S2,S2,Z1Z1);
    fe_sub(H,U2,P.X);
    fe_sub(t,S2,P.Y); fe_add(r,t,t);
    if (is_zero(H)) { if (is_zero(r)) return jac_dbl(P); return Jac{ZERO,ZERO,ZERO,true}; }
    fe_sqr(HH,H); fe_add(I,HH,HH); fe_add(I,I,I);
    fe_mul(J,H,I); fe_mul(V,P.X,I);
    Jac R; R.inf=false;
    fe_sqr(R.X,r); fe_sub(R.X,R.X,J); fe_sub(R.X,R.X,V); fe_sub(R.X,R.X,V);
    fe_sub(t,V,R.X); fe_mul(t,r,t);
    U256 u; fe_mul(u,P.Y,J); fe_add(u,u,u);
    fe_sub(R.Y,t,u);
    fe_add(t,P.Z,H); fe_sqr(t,t); fe_sub(t,t,Z1Z1); fe_sub(R.Z,t,HH);
    return R;
}
static Jac jac_mul(const U256 &k, const Point &P) {
    Jac R; R.X=R.Y=R.Z=ZERO; R.inf=true;
    for (int i = highest_bit(k); i >= 0; --i) { R = jac_dbl(R); if (get_bit(k,i)) R = jac_madd(R,P); }
    return R;
}
static Point jac_to_aff(const Jac &J) {
    Point A; A.x=ZERO; A.y=ZERO;
    if (J.inf) { A.inf=true; return A; }
    U256 zi=fe_inv(J.Z), zi2, zi3;
    fe_sqr(zi2,zi); fe_mul(zi3,zi2,zi);
    fe_mul(A.x,J.X,zi2); fe_mul(A.y,J.Y,zi3); A.inf=false;
    return A;
}

// =================== Table de multiplication à base fixe (calcul rapide de k*G) ===================
// 16 fenêtres de 8 bits ; chaque fenêtre contient les 255 multiples non nuls de 2^(8w)*G.
// Construite une seule fois au démarrage (4080 additions affines, ~20 ms) ; scalar_mul_fixed
// combine ensuite au plus 16 points par addition Jacobienne mixte (une seule inversion finale).
static const int GT_W = 8, GT_BITS = 128, GT_WINDOWS = GT_BITS / GT_W, GT_SIZE = 1 << GT_W;
static std::vector<Point> g_gtable;

static std::vector<Point> build_fixed_table(const Point &g) {
    std::vector<Point> table((size_t)GT_WINDOWS * GT_SIZE, Point{ZERO, ZERO, true});
    Point base = g;
    for (int w = 0; w < GT_WINDOWS; w++) {
        Point cur = base;
        table[(size_t)w*GT_SIZE + 1] = cur;
        for (int d = 2; d < GT_SIZE; d++) { cur = aff_add(cur, base); table[(size_t)w*GT_SIZE + d] = cur; }
        base = aff_add(cur, base);
    }
    return table;
}
static void init_gtable(const Point &g) { g_gtable = build_fixed_table(g); }
static Point scalar_mul_fixed(const U256 &k) {
    Jac acc; acc.X=acc.Y=acc.Z=ZERO; acc.inf=true;
    for (int w = 0; w < GT_WINDOWS; w++) {
        int pos = w * GT_W, limb = pos / 64;
        if (limb >= 4) break;
        unsigned d = (unsigned)((k.d[limb] >> (pos % 64)) & (GT_SIZE - 1));
        if (d) acc = jac_madd(acc, g_gtable[(size_t)w*GT_SIZE + d]);
    }
    return jac_to_aff(acc);
}

// =================== SHA-256 (un seul bloc, messages <= 55 octets) ===================
static inline u32 rotr32(u32 x,int n){ return (x>>n)|(x<<(32-n)); }
static inline u32 rotl32(u32 x,int n){ return (x<<n)|(x>>(32-n)); }
static const u32 SHA_K[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
static inline void sha256_block(const u32* w16, u32* out) {
    u32 w[64];
    for (int i=0;i<16;i++) w[i]=w16[i];
    for (int i=16;i<64;i++) {
        u32 s0=rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
        u32 s1=rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    u32 a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
    for (int i=0;i<64;i++) {
        u32 S1=rotr32(e,6)^rotr32(e,11)^rotr32(e,25), ch=(e&f)^(~e&g);
        u32 t1=h+S1+ch+SHA_K[i]+w[i];
        u32 S0=rotr32(a,2)^rotr32(a,13)^rotr32(a,22), mj=(a&b)^(a&c)^(b&c);
        u32 t2=S0+mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    out[0]=a+0x6a09e667; out[1]=b+0xbb67ae85; out[2]=c+0x3c6ef372; out[3]=d+0xa54ff53a;
    out[4]=e+0x510e527f; out[5]=f+0x9b05688c; out[6]=g+0x1f83d9ab; out[7]=h+0x5be0cd19;
}
static void sha256_bytes(const unsigned char* m, size_t n, unsigned char* out32) {
    unsigned char buf[64]; memset(buf,0,64); memcpy(buf,m,n); buf[n]=0x80;
    u64 bl=(u64)n*8; for (int i=0;i<8;i++) buf[63-i]=(unsigned char)(bl>>(8*i));
    u32 w[16],o[8];
    for (int i=0;i<16;i++) w[i]=((u32)buf[4*i]<<24)|((u32)buf[4*i+1]<<16)|((u32)buf[4*i+2]<<8)|buf[4*i+3];
    sha256_block(w,o);
    for (int i=0;i<8;i++){ out32[4*i]=o[i]>>24; out32[4*i+1]=o[i]>>16; out32[4*i+2]=o[i]>>8; out32[4*i+3]=o[i]; }
}
static void sha256d(const unsigned char* d, size_t n, unsigned char out[32]) {
    unsigned char t[32]; sha256_bytes(d,n,t); sha256_bytes(t,32,out);
}

// =================== RIPEMD-160 (un seul bloc, scalaire) ===================
static const unsigned char RMD_R1[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2, 4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
static const unsigned char RMD_R2[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14, 12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
static const unsigned char RMD_S1[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12, 9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
static const unsigned char RMD_S2[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8, 8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
static const u32 RMD_K1[5]={0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
static const u32 RMD_K2[5]={0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};
template<int F> static inline u32 rmd_f(u32 x,u32 y,u32 z){
    if(F==0) return x^y^z;
    if(F==1) return (x&y)|(~x&z);
    if(F==2) return (x|~y)^z;
    if(F==3) return (x&z)|(y&~z);
    return x^(y|~z);
}
static inline void rmd160_block(const u32* X, u32* out) {
    u32 a=0x67452301,b=0xEFCDAB89,c=0x98BADCFE,d=0x10325476,e=0xC3D2E1F0;
    u32 a2=a,b2=b,c2=c,d2=d,e2=e;
    auto seg=[&](int F){
        for (int j=16*F;j<16*F+16;j++) {
            u32 t = rotl32(a + (F==0?rmd_f<0>(b,c,d):F==1?rmd_f<1>(b,c,d):F==2?rmd_f<2>(b,c,d):F==3?rmd_f<3>(b,c,d):rmd_f<4>(b,c,d)) + X[RMD_R1[j]] + RMD_K1[F], RMD_S1[j]) + e;
            a=e; e=d; d=rotl32(c,10); c=b; b=t;
            int G2 = 4-F;
            u32 t2 = rotl32(a2 + (G2==0?rmd_f<0>(b2,c2,d2):G2==1?rmd_f<1>(b2,c2,d2):G2==2?rmd_f<2>(b2,c2,d2):G2==3?rmd_f<3>(b2,c2,d2):rmd_f<4>(b2,c2,d2)) + X[RMD_R2[j]] + RMD_K2[F], RMD_S2[j]) + e2;
            a2=e2; e2=d2; d2=rotl32(c2,10); c2=b2; b2=t2;
        }
    };
    seg(0); seg(1); seg(2); seg(3); seg(4);
    u32 t=0xEFCDAB89u+c+d2;
    out[1]=0x98BADCFEu+d+e2; out[2]=0x10325476u+e+a2; out[3]=0xC3D2E1F0u+a+b2; out[4]=0x67452301u+b+c2; out[0]=t;
}
static void hash160_bytes(const unsigned char* m, size_t n, unsigned char* out20) {
    unsigned char dg[32]; sha256_bytes(m,n,dg);
    u32 X[16]={0},o[5];
    for (int i=0;i<8;i++) X[i]=(u32)dg[4*i]|((u32)dg[4*i+1]<<8)|((u32)dg[4*i+2]<<16)|((u32)dg[4*i+3]<<24);
    X[8]=0x80; X[14]=256;
    rmd160_block(X,o);
    for (int i=0;i<5;i++){ out20[4*i]=o[i]; out20[4*i+1]=o[i]>>8; out20[4*i+2]=o[i]>>16; out20[4*i+3]=o[i]>>24; }
}
static U256 pack20(const unsigned char* b) {
    U256 r=ZERO; memcpy(&r.d[0],b,8); memcpy(&r.d[1],b+8,8); u32 t; memcpy(&t,b+16,4); r.d[2]=t; return r;
}

#ifndef __CUDACC__
// ---- hash160 vectorise (8 cles a la fois, SIMD) : moteur CPU uniquement ----
// ---- hash160 vectorisé (8 clés à la fois) : le cœur de l'accélération de ce moteur ----
#ifndef VLANES
#define VLANES 8
#endif
typedef u32 vu __attribute__((vector_size(4*VLANES)));
static inline vu vsplat(u32 x){ vu v; for(int i=0;i<VLANES;i++) v[i]=x; return v; }
#define VROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define VROTL(x,n) (((x)<<(n))|((x)>>(32-(n))))
template<int F> static inline vu rmd_fv(vu x,vu y,vu z){
    if(F==0) return x^y^z;
    if(F==1) return (x&y)|(~x&z);
    if(F==2) return (x|~y)^z;
    if(F==3) return (x&z)|(y&~z);
    return x^(y|~z);
}
template<int F> static inline void rmd_segv(const vu* X,vu& a,vu& b,vu& c,vu& d,vu& e,vu& a2,vu& b2,vu& c2,vu& d2,vu& e2){
    for (int j=16*F;j<16*F+16;j++) {
        vu t=VROTL(a+rmd_fv<F>(b,c,d)+X[RMD_R1[j]]+vsplat(RMD_K1[F]),RMD_S1[j])+e;
        a=e; e=d; d=VROTL(c,10); c=b; b=t;
        t=VROTL(a2+rmd_fv<4-F>(b2,c2,d2)+X[RMD_R2[j]]+vsplat(RMD_K2[F]),RMD_S2[j])+e2;
        a2=e2; e2=d2; d2=VROTL(c2,10); c2=b2; b2=t;
    }
}
// hash160 de VLANES clés publiques compressées (xs[l], pars[l]) -> out[l] (empaqueté comme pack20)
static void hash160_vec(const U256* xs, const u64* pars, U256* out) {
    u32 Xl[8][VLANES], Pl[VLANES];
    for (int l=0;l<VLANES;l++) {
        const U256& x = xs[l];
        Xl[0][l]=(u32)(x.d[3]>>32); Xl[1][l]=(u32)x.d[3]; Xl[2][l]=(u32)(x.d[2]>>32); Xl[3][l]=(u32)x.d[2];
        Xl[4][l]=(u32)(x.d[1]>>32); Xl[5][l]=(u32)x.d[1]; Xl[6][l]=(u32)(x.d[0]>>32); Xl[7][l]=(u32)x.d[0];
        Pl[l]=0x02u+(u32)pars[l];
    }
    vu X[8],w[64],H[8],rw[16];
    for (int i=0;i<8;i++) memcpy(&X[i],Xl[i],sizeof(vu));
    vu P; memcpy(&P,Pl,sizeof(vu));
    w[0]=(P<<24)|(X[0]>>8);
    for (int i=1;i<8;i++) w[i]=(X[i-1]<<24)|(X[i]>>8);
    w[8]=(X[7]<<24)|vsplat(0x00800000u);
    for (int i=9;i<15;i++) w[i]=vsplat(0);
    w[15]=vsplat(264);
    for (int i=16;i<64;i++) {
        vu s0=VROTR(w[i-15],7)^VROTR(w[i-15],18)^(w[i-15]>>3);
        vu s1=VROTR(w[i-2],17)^VROTR(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    vu a=vsplat(0x6a09e667u),b=vsplat(0xbb67ae85u),c=vsplat(0x3c6ef372u),d=vsplat(0xa54ff53au);
    vu e=vsplat(0x510e527fu),f=vsplat(0x9b05688cu),g=vsplat(0x1f83d9abu),h=vsplat(0x5be0cd19u);
    for (int i=0;i<64;i++) {
        vu S1=VROTR(e,6)^VROTR(e,11)^VROTR(e,25), ch=(e&f)^(~e&g);
        vu t1=h+S1+ch+vsplat(SHA_K[i])+w[i];
        vu S0=VROTR(a,2)^VROTR(a,13)^VROTR(a,22), mj=(a&b)^(a&c)^(b&c);
        vu t2=S0+mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    H[0]=a+vsplat(0x6a09e667u); H[1]=b+vsplat(0xbb67ae85u); H[2]=c+vsplat(0x3c6ef372u); H[3]=d+vsplat(0xa54ff53au);
    H[4]=e+vsplat(0x510e527fu); H[5]=f+vsplat(0x9b05688cu); H[6]=g+vsplat(0x1f83d9abu); H[7]=h+vsplat(0x5be0cd19u);
    for (int i=0;i<8;i++) {
        vu q=H[i];
        rw[i]=(q>>24)|((q>>8)&vsplat(0x0000FF00u))|((q<<8)&vsplat(0x00FF0000u))|(q<<24);
    }
    rw[8]=vsplat(0x80); for(int i=9;i<14;i++) rw[i]=vsplat(0); rw[14]=vsplat(256); rw[15]=vsplat(0);
    vu a1=vsplat(0x67452301u),b1=vsplat(0xEFCDAB89u),c1=vsplat(0x98BADCFEu),d1=vsplat(0x10325476u),e1=vsplat(0xC3D2E1F0u);
    vu a2=a1,b2=b1,c2=c1,d2=d1,e2=e1;
    rmd_segv<0>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<1>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<2>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<3>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<4>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    vu o[5];
    o[0]=vsplat(0xEFCDAB89u)+c1+d2; o[1]=vsplat(0x98BADCFEu)+d1+e2; o[2]=vsplat(0x10325476u)+e1+a2;
    o[3]=vsplat(0xC3D2E1F0u)+a1+b2; o[4]=vsplat(0x67452301u)+b1+c2;
    u32 T[5][VLANES];
    for (int k=0;k<5;k++) memcpy(T[k],&o[k],sizeof(vu));
    for (int l=0;l<VLANES;l++) {
        out[l].d[0]=T[0][l]|((u64)T[1][l]<<32); out[l].d[1]=T[2][l]|((u64)T[3][l]<<32);
        out[l].d[2]=T[4][l]; out[l].d[3]=0;
    }
}
#endif // !__CUDACC__

// =================== Base58Check (adresse P2PKH) et WIF ===================
static const char *B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static bool decode_address(const std::string &addr, unsigned char out[20]) {
    unsigned char buf[25] = {0};
    for (char c : addr) {
        const char *p = c ? strchr(B58, c) : nullptr;
        if (!p) return false;
        int carry = p - B58;
        for (int i = 24; i >= 0; --i) { carry += 58 * buf[i]; buf[i] = carry & 0xFF; carry >>= 8; }
        if (carry) return false;
    }
    unsigned char h[32]; sha256d(buf, 21, h);
    if (memcmp(h, buf + 21, 4) != 0 || buf[0] != 0x00) return false;
    memcpy(out, buf + 1, 20);
    return true;
}
static std::string base58_encode(const std::vector<unsigned char> &in) {
    std::vector<unsigned char> d(in.size() * 138 / 100 + 1, 0);
    size_t len = 0;
    for (unsigned char b : in) {
        int carry = b; size_t i = 0;
        for (auto it = d.rbegin(); (carry || i < len) && it != d.rend(); ++it, ++i) {
            carry += 256 * (*it); *it = carry % 58; carry /= 58;
        }
        len = i;
    }
    auto it = d.begin(); while (it != d.end() && *it == 0) ++it;
    std::string s(std::count(in.begin(), in.begin() + std::min<size_t>(1, in.size()), 0), '1');
    for (; it != d.end(); ++it) s += B58[*it];
    return s;
}
static void to_bytes32_priv(u128 k, unsigned char out[32]) {
    memset(out, 0, 32);
    for (int i = 31; i >= 16; --i) { out[i] = (unsigned char)(k & 0xFF); k >>= 8; }
}
static std::string to_hex(u128 k) {
    char t[40]; snprintf(t, sizeof t, "%016llx%016llx", (unsigned long long)(k>>64), (unsigned long long)k);
    std::string s(t); size_t p = s.find_first_not_of('0');
    return "0x" + (p == std::string::npos ? "0" : s.substr(p));
}
// Nombre de bits nécessaires pour représenter v (0 pour v == 0, 1 pour v == 1, etc.).
static int u128_bitlen(u128 v) {
    int n = 0;
    while (v) { ++n; v >>= 1; }
    return n;
}
static uint64_t next_pow2_u64(uint64_t x) {
    uint64_t p = 1;
    while (p < x) p <<= 1;
    return p;
}
// Parse une valeur hexadécimale (préfixe "0x"/"0X" optionnel) dans un u128. Retourne false si la
// chaîne est vide, contient un caractère non hexadécimal, ou dépasse 128 bits (32 chiffres hex).
static bool parse_u128_hex(const std::string &in, u128 &out) {
    std::string s = in;
    if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s = s.substr(2);
    if (s.empty() || s.size() > 32) return false;
    u128 v = 0;
    for (char c : s) {
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (u128)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (u128)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (u128)(c - 'A' + 10);
        else return false;
    }
    out = v;
    return true;
}
// Parse un argument de plage "début:fin" en hexadécimal. Retourne false si le format est invalide.
static bool parse_hex_range(const std::string &arg, u128 &lo, u128 &hi) {
    size_t p = arg.find(':');
    if (p == std::string::npos) return false;
    u128 a, b;
    if (!parse_u128_hex(arg.substr(0, p), a) || !parse_u128_hex(arg.substr(p + 1), b)) return false;
    lo = std::min(a, b); hi = std::max(a, b);
    return true;
}
static std::string to_wif(u128 k) {
    std::vector<unsigned char> v(38);
    v[0] = 0x80; to_bytes32_priv(k, &v[1]); v[33] = 0x01;
    unsigned char h[32]; sha256d(v.data(), 34, h); memcpy(&v[34], h, 4);
    return base58_encode(v);
}

// =================== Moteur de scan ===================
static std::atomic<bool> g_stop{false}, g_found{false};
static std::atomic<uint64_t> g_count{0};
static std::mutex g_mtx, g_file_mtx, g_print_mtx; // g_print_mtx protège stdout (progression vs "CLE TROUVEE")

// Ensemble de cibles (hash160). Un puzzle seul = 1 cible ; -pApB = jusqu'à 80 cibles cherchées
// simultanément pendant le MÊME balayage de l'intervalle commun. Un filtre binaire de 65536 bits
// (8 Ko, reste dans le cache L1) indexé par les 16 bits de poids faible du hash160 écarte d'emblée
// ~99,9 % des clés (80 cibles -> 0,12 % de faux positifs) avant toute comparaison complète.
static const int MAX_TARGETS = 80;
struct TargetSet {
    std::vector<U256> h;           // hash160 empaquetés (pack20)
    std::vector<std::string> addr; // adresse P2PKH de chaque cible
    std::vector<int> puzzle;       // numéro de puzzle (0 = adresse libre hors puzzle)
    u64 filter[1024] = {};
    void add(const std::string &a, const unsigned char h20[20], int pz) {
        h.push_back(pack20(h20)); addr.push_back(a); puzzle.push_back(pz);
        u64 ix = h.back().d[0] & 0xFFFF; filter[ix >> 6] |= 1ULL << (ix & 63);
    }
    size_t size() const { return h.size(); }
    // Indice de la cible égale à x, ou -1.
    inline int find(const U256 &x) const {
        u64 ix = x.d[0] & 0xFFFF;
        if (!((filter[ix >> 6] >> (ix & 63)) & 1)) return -1;
        for (size_t i = 0; i < h.size(); ++i) if (eq(h[i], x)) return (int)i;
        return -1;
    }
};
static TargetSet g_ts;
static std::atomic<bool> g_tfound[MAX_TARGETS]; // cible déjà trouvée (évite les doublons d'enregistrement)
static std::atomic<int>  g_remaining{0};        // nombre de cibles restant à trouver

static uint64_t g_next = 0, g_nchunks = 0;
static std::vector<uint64_t> g_cur;
static const uint64_t IDLE = UINT64_MAX;

static bool claim(unsigned t, uint64_t &idx) {
    std::lock_guard<std::mutex> l(g_mtx);
    if (g_next >= g_nchunks) return false;
    idx = g_next++; g_cur[t] = idx; return true;
}
static void release(unsigned t) { std::lock_guard<std::mutex> l(g_mtx); g_cur[t] = IDLE; }
static uint64_t safe_index() {
    std::lock_guard<std::mutex> l(g_mtx);
    uint64_t m = g_next; for (auto c : g_cur) m = std::min(m, c); return m;
}

// Sérialise la coordonnée X en 32 octets big-endian (SEC1).
static inline void x_to_bytes(const U256 &x, unsigned char out[32]) {
    for (int i = 0; i < 4; i++) {
        u64 limb = x.d[3 - i];
        for (int j = 0; j < 8; j++) out[i*8 + j] = (unsigned char)(limb >> (56 - 8*j));
    }
}
// Sérialise un point affine en clé publique compressée SEC1 (33 octets : préfixe de parité + X).
static inline void point_to_compressed(const Point &p, unsigned char out[33]) {
    out[0] = (p.y.d[0] & 1) ? 0x03 : 0x02;
    x_to_bytes(p.x, out + 1);
}
static bool verify_key(u128 k, int ti) {
    Point p = jac_to_aff(jac_mul(u128_to_U256(k), G));
    unsigned char comp[33], h[20];
    point_to_compressed(p, comp);
    hash160_bytes(comp, 33, h);
    return eq(pack20(h), g_ts.h[ti]);
}
static void save_key(u128 k, int ti) {
    const std::string &addr = g_ts.addr[ti];
    int pz = g_ts.puzzle[ti];
    std::string hex_k = to_hex(k);
    std::string wif_k = to_wif(k);
    std::lock_guard<std::mutex> l(g_file_mtx);
    std::ofstream f("God.txt", std::ios::app);
    f << "Adresse : " << addr << "\n";
    if (pz > 0) f << "Puzzle : #" << pz << "\n";
    f << "Clé privée (hex) : " << hex_k << "\n"
      << "WIF (compressé)  : " << wif_k << "\n"
      << "----------------------------------------\n";
    f.flush();
    {
        std::lock_guard<std::mutex> lp(g_print_mtx);
        printf("\n\n*** CLE TROUVEE ***\n");
        if (pz > 0) printf("Puzzle           : #%d\n", pz);
        printf("Adresse          : %s\n", addr.c_str());
        printf("Cle privee (hex) : %s\n", hex_k.c_str());
        printf("WIF (compresse)  : %s\n", wif_k.c_str());
        printf("Enregistree dans God.txt\n");
        fflush(stdout);
    }
}
// Traite une correspondance signalée par scan_batch : vérifie la clé en recalculant tout depuis
// zéro, l'enregistre UNE seule fois, et ne déclenche l'arrêt général que quand TOUTES les cibles
// sont trouvées (g_found == « toutes trouvées »).
static void handle_hit(u128 k, int ti) {
    if (!verify_key(k, ti)) return;
    if (g_tfound[ti].exchange(true)) return; // déjà trouvée (ex: tirage aléatoire qui repasse dessus)
    save_key(k, ti);
    if (g_remaining.fetch_sub(1) == 1) { g_found = true; g_stop = true; }
}
// Gestionnaire de signal (Ctrl-C / SIGTERM) : ne fait que des opérations async-signal-safe
// (std::atomic lock-free + write()), jamais de printf/cout/mutex, qui ne sont pas garantis
// sûrs à appeler depuis un gestionnaire de signal et pourraient bloquer (deadlock) s'il est
// déclenché pendant que le thread principal tient déjà le verrou d'affichage.
// 1er Ctrl-C  : demande un arrêt propre (g_stop) — les threads finissent leur segment en cours
//               (quelques centaines de µs à quelques ms au plus, vérifié très fréquemment) puis
//               sont rejoints normalement, et le résumé final est affiché.
// 2e Ctrl-C   : au cas où l'arrêt propre ne répondrait pas (ex: calcul bloqué), sortie immédiate.
static std::atomic<int> g_signal_count{0};
static void on_signal(int) {
    g_stop = true;
    if (g_signal_count.fetch_add(1) == 0) {
        static const char msg[] =
            "\n[Ctrl-C] Arret demande : fin du segment en cours puis sortie propre "
            "(Ctrl-C a nouveau pour forcer l'arret immediat)...\n";
        ssize_t r = write(STDOUT_FILENO, msg, sizeof(msg) - 1); (void)r;
    } else {
        static const char msg2[] = "\n[Ctrl-C] Arret force.\n";
        ssize_t r = write(STDOUT_FILENO, msg2, sizeof(msg2) - 1); (void)r;
        _exit(130);
    }
}
static double g_display_interval = 30.0; // secondes entre deux rafraîchissements de l'affichage (-s T)

// Formate un nombre de clés (ou de clés/s) avec l'unité adaptée : k, M, G, T, P.
static std::string fmt_keys(double v) {
    static const char *U[] = {"", "k", "M", "G", "T", "P"};
    int i = 0;
    double a = v;
    while (a >= 1000.0 && i < 5) { a /= 1000.0; ++i; }
    char b[32];
    if (i == 0) snprintf(b, sizeof b, "%.0f", a);
    else snprintf(b, sizeof b, "%.2f %s", a, U[i]);
    return b;
}
static std::string fmt_time(double s) {
    if (s > 3.15e9) return ">100 ans";
    char b[64];
    if (s > 3.15e7) snprintf(b, sizeof b, "%.1f ans", s / 3.15e7);
    else if (s > 86400) snprintf(b, sizeof b, "%.1f j", s / 86400);
    else snprintf(b, sizeof b, "%02d:%02d:%02d", (int)s/3600, ((int)s/60)%60, (int)s%60);
    return b;
}

// Fragment d'affichage « | trouvées X/N » (vide quand il n'y a qu'une seule cible).
static std::string fmt_targets() {
    int n = (int)g_ts.size();
    if (n <= 1) return "";
    return " | trouvées " + std::to_string(n - g_remaining.load()) + "/" + std::to_string(n);
}

#ifndef __CUDACC__
// =================== Fenêtre ±i centrée : le cœur de la vitesse ===================
// Table des multiples 1..HALF du point de pas 'base' (G pour le mode par défaut, un point
// quelconque pour le mode voisinage), plus le pas vers le lot suivant (2*HALF*base).
static const int HALF = 512;
struct WinTable { std::vector<U256> gx, gy; U256 sx, sy; };

static void build_wintable(WinTable &wt, const Point &base) {
    wt.gx.assign(HALF + 1, ZERO); wt.gy.assign(HALF + 1, ZERO);

    // Accumule 1*base .. HALF*base, plus le point de pas suivant 2*HALF*base, en coordonnées de
    // Jacobi (jac_madd / jac_dbl : aucune inversion par point), puis convertit tout le lot en
    // affine avec UNE SEULE inversion (même astuce de lot que scan_batch), au lieu des HALF
    // inversions individuelles que faisait l'ancienne version via aff_add. C'est ce qui rendait
    // build_wintable très coûteuse en mode voisinage, où elle est reconstruite à chaque segment.
    const int M = HALF + 1; // +1 pour le point de pas suivant
    // Buffers réutilisés d'un appel à l'autre : build_wintable est rappelée à chaque segment en
    // mode voisinage, donc réallouer 4 vecteurs à chaque appel serait un coût fixe inutile.
    // Pas de risque de concurrence : build_wintable n'est jamais appelée depuis plusieurs threads
    // à la fois (une fois au démarrage, sinon séquentiellement depuis la boucle d'essais).
    static std::vector<U256> X(M + 1), Y(M + 1), Z(M + 1), pre(M + 1); // indices 1..M

    Jac cur; cur.X = base.x; cur.Y = base.y; cur.Z = ONE; cur.inf = false; // 1*base
    X[1] = cur.X; Y[1] = cur.Y; Z[1] = cur.Z;
    for (int i = 2; i <= HALF; i++) {
        cur = jac_madd(cur, base);
        X[i] = cur.X; Y[i] = cur.Y; Z[i] = cur.Z;
    }
    Jac s = jac_dbl(cur); // cur == HALF*base ici, donc s == 2*HALF*base
    X[M] = s.X; Y[M] = s.Y; Z[M] = s.Z;

    // Inversion de lot : produit préfixe, une seule fe_inv, puis répartition arrière.
    U256 acc = ONE;
    for (int i = 1; i <= M; i++) { fe_mul(acc, acc, Z[i]); pre[i] = acc; }
    U256 inv = fe_inv(acc);
    for (int i = M; i >= 1; --i) {
        U256 zi; // 1/Z[i]
        if (i > 1) { fe_mul(zi, inv, pre[i-1]); fe_mul(inv, inv, Z[i]); }
        else zi = inv;
        U256 zi2, zi3, ax, ay;
        fe_sqr(zi2, zi); fe_mul(zi3, zi2, zi);
        fe_mul(ax, X[i], zi2); fe_mul(ay, Y[i], zi3);
        if (i <= HALF) { wt.gx[i] = ax; wt.gy[i] = ay; } else { wt.sx = ax; wt.sy = ay; }
    }
}

struct Scratch {
    std::vector<U256> pre, cx; std::vector<char> zero; std::vector<u64> cpar; std::vector<int> coff;
    Scratch() : pre(HALF+2), cx(2*HALF+2*VLANES+2), zero(HALF+2), cpar(2*HALF+2*VLANES+2), coff(2*HALF+2*VLANES+2) {}
};

// Scanne N pas consécutifs à partir du point d'indice 0 'start', le long de la table 'wt'.
// Appelle on_hit(offset, indice_cible) pour CHAQUE clé dont le hash160 correspond à l'une des
// cibles de 'ts' (offset = position de la clé par rapport à 'start'), sans interrompre le balayage :
// indispensable quand plusieurs cibles sont cherchées à la fois. Retourne -1 si le balayage est
// terminé, -2 s'il a été interrompu par should_stop.
template <typename StopFn, typename HitFn>
static int64_t scan_batch(const WinTable &wt, const Point &start, uint64_t N, Scratch &sc,
                           std::atomic<uint64_t> &counter, const TargetSet &ts,
                           StopFn should_stop, HitFn on_hit) {
    Point C = aff_add(start, Point{wt.gx[HALF], wt.gy[HALF], false}); // point d'indice HALF
    uint64_t done = 0;
    while (done < N) {
        if (should_stop()) return -2;
        uint64_t count = std::min<uint64_t>((uint64_t)(2*HALF), N - done);
        bool last = count < (uint64_t)(2*HALF);
        int nc = 0;
        auto emit = [&](const U256 &x, u64 par, int off) { sc.cx[nc]=x; sc.cpar[nc]=par; sc.coff[nc]=off; nc++; };
        emit(C.x, C.y.d[0] & 1, 0);

        U256 acc = ONE;
        for (int i = 1; i <= HALF; i++) {
            U256 d; fe_sub(d, wt.gx[i], C.x);
            if (is_zero(d)) { sc.zero[i] = 1; d = ONE; } else sc.zero[i] = 0;
            fe_mul(acc, acc, d); sc.pre[i] = acc;
        }
        U256 dS; fe_sub(dS, wt.sx, C.x);
        bool zs = is_zero(dS); if (zs) dS = ONE;
        fe_mul(acc, acc, dS);
        U256 inv = fe_inv(acc);

        Point Cn{ZERO, ZERO, false};
        {
            U256 invd; fe_mul(invd, inv, sc.pre[HALF]); fe_mul(inv, inv, dS);
            if (!zs && !last) {
                U256 t, l; fe_sub(t, wt.sy, C.y); fe_mul(l, t, invd);
                fe_sqr(Cn.x, l); fe_sub(Cn.x, Cn.x, C.x); fe_sub(Cn.x, Cn.x, wt.sx);
                fe_sub(t, C.x, Cn.x); fe_mul(Cn.y, l, t); fe_sub(Cn.y, Cn.y, C.y);
            }
        }
        for (int i = HALF; i >= 1; --i) {
            U256 invd;
            if (i > 1) {
                fe_mul(invd, inv, sc.pre[i-1]);
                U256 d; fe_sub(d, wt.gx[i], C.x); if (sc.zero[i]) d = ONE;
                fe_mul(inv, inv, d);
            } else invd = inv;
            if (sc.zero[i]) { // cas dégénéré (rare) : repli sur l'addition affine directe
                U256 ny; fe_sub(ny, ZERO, wt.gy[i]);
                Point pm = aff_add(C, Point{wt.gx[i], ny, false});
                emit(pm.x, pm.y.d[0] & 1, -i);
                if (i < HALF) { Point pp = aff_add(C, Point{wt.gx[i], wt.gy[i], false}); emit(pp.x, pp.y.d[0] & 1, i); }
                continue;
            }
            U256 t, l, x3, y3;
            fe_add(t, wt.gy[i], C.y); fe_mul(l, t, invd);
            fe_sqr(x3, l); fe_sub(x3, x3, C.x); fe_sub(x3, x3, wt.gx[i]);
            fe_sub(t, x3, C.x); fe_mul(y3, l, t); fe_sub(y3, y3, C.y);
            emit(x3, y3.d[0] & 1, -i);
            if (i < HALF) {
                fe_sub(t, wt.gy[i], C.y); fe_mul(l, t, invd);
                fe_sqr(x3, l); fe_sub(x3, x3, C.x); fe_sub(x3, x3, wt.gx[i]);
                fe_sub(t, C.x, x3); fe_mul(y3, l, t); fe_sub(y3, y3, C.y);
                emit(x3, y3.d[0] & 1, i);
            }
        }

        for (int c = 0; c < nc; c += VLANES) {
            U256 xs[VLANES], hs[VLANES]; u64 ps[VLANES]; int m = std::min((int)VLANES, nc - c);
            for (int l = 0; l < VLANES; l++) { int idx = c + (l < m ? l : m - 1); xs[l] = sc.cx[idx]; ps[l] = sc.cpar[idx]; }
            hash160_vec(xs, ps, hs);
            for (int l = 0; l < m; l++) {
                int ti = ts.find(hs[l]);
                if (ti >= 0) on_hit((int64_t)done + HALF + sc.coff[c+l], ti);
            }
        }
        counter.fetch_add(count, std::memory_order_relaxed);
        done += count;
        if (last) break;
        if (zs) C = aff_add(C, Point{wt.sx, wt.sy, false}); else C = Cn;
    }
    return -1;
}

static WinTable g_wt_main; // table construite à partir de G, pour le mode par défaut (pas = 1 clé)

// lo/span délimitent la plage réelle à couvrir (span = fin - début + 1, pas nécessairement une
// puissance de deux pour une plage personnalisée "début:fin" en hexadécimal). g_nchunks reste une
// puissance de deux (nécessaire pour que le mélange par 'stride' soit une bijection sans doublon) :
// certains indices permutés tombent donc au-delà de span (reliquat de l'arrondi à la puissance de
// deux supérieure) et sont simplement sautés ; le dernier chunk réel, lui, est tronqué à la taille
// qu'il lui reste dans la plage.
static void worker(unsigned t, u128 lo, u128 span, int cb, uint64_t stride) {
    const uint64_t n_full = 1ULL << cb, mask = g_nchunks - 1;
    uint64_t idx;
    Scratch sc;
    while (!g_stop.load(std::memory_order_relaxed) && claim(t, idx)) {
        u128 offset = (u128)((idx * stride) & mask) << cb;
        if (offset >= span) { release(t); continue; } // chunk hors plage réelle : rien à scanner
        uint64_t n = (uint64_t)std::min<u128>((u128)n_full, span - offset);
        u128 start = lo + offset;
        Point start_pt = scalar_mul_fixed(u128_to_U256(start));
        bool finished = true;
        scan_batch(g_wt_main, start_pt, n, sc, g_count, g_ts, [&]() {
            if (g_stop.load(std::memory_order_relaxed)) { finished = false; return true; }
            return false;
        }, [&](int64_t hit, int ti) { handle_hit(start + (u128)hit, ti); });
        if (finished) release(t); else return;
    }
}
#endif // !__CUDACC__

struct Segment { int start_bit; int len; const char *nom; };
static bool segment_fits(int free_bits, int N) { return N >= 1 && N <= free_bits; }
static std::vector<Segment> build_segments(int level, int free_bits, int N) {
    Segment prefix{ free_bits - N, N, "prefixe" };
    Segment suffix{ 0, N, "suffixe" };
    Segment middle{ (free_bits - N) / 2, N, "milieu" };
    switch (level) {
        case 1: return {prefix};
        case 2: return {middle};
        case 3: return {suffix};
        default: return {prefix, middle, suffix};
    }
}
struct CellPerm {
    static const int ROUNDS = 6;
    int k = 0, h = 0; u128 total = 1; u64 mask = 0; u64 rk[ROUNDS] = {};
    static inline u64 mix(u64 z) {
        z += 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    void init(int kk, std::mt19937_64 &g) {
        k = kk; total = (u128)1 << k; h = (k + 1) / 2;
        mask = (h >= 64) ? ~0ULL : ((1ULL << h) - 1);
        for (int r = 0; r < ROUNDS; r++) rk[r] = g();
    }
    u128 apply(u128 x) const {
        if (k == 0) return 0;
        do {
            u64 L = (u64)(x >> h) & mask, R = (u64)x & mask;
            for (int r = 0; r < ROUNDS; r++) { u64 f = L ^ (mix(R ^ rk[r]) & mask); L = R; R = f; }
            x = ((u128)L << h) | R;
        } while (x >= total);
        return x;
    }
};
static inline u128 cell_to_off(u128 c, const Segment &seg) {
    u128 low = c & (((u128)1 << seg.start_bit) - 1);
    return ((c >> seg.start_bit) << (seg.start_bit + seg.len)) | low;
}


#ifndef __CUDACC__
static std::atomic<uint64_t> g_nb_next{0};      // prochain indice de cellule à réserver (= essais lancés)
static std::atomic<bool> g_nb_abandoned{false}; // un thread a quitté un essai en cours (arrêt/Ctrl-C/clé trouvée)

// Corps d'un thread du mode voisinage. Chaque thread réserve l'indice suivant (fetch_add, sans
// verrou), en déduit sa cellule par la permutation, puis balaie TOUT SEUL les segments de cet essai
// avant de passer au suivant : les nt threads travaillent sur des essais indépendants en parallèle,
// donc tous les cœurs restent occupés quelle que soit la taille des segments.
// Aux niveaux -vvvv (3 segments) le même motif c sert aux trois segments (N bits insérés en préfixe,
// au milieu puis en suffixe) : chaque type de segment parcourt ainsi lui aussi l'intervalle sans doublon.
static void neighbor_worker(const CellPerm &perm, u128 total_cells, u128 lo, u128 span,
                             const std::vector<Segment> &segments, const std::vector<WinTable> &seg_wt) {
    Scratch sc;
    auto must_stop = []() { return g_stop.load(std::memory_order_relaxed) || g_found.load(std::memory_order_relaxed); };
    while (!must_stop()) {
        uint64_t idx = g_nb_next.fetch_add(1, std::memory_order_relaxed);
        if ((u128)idx >= total_cells) break; // toutes les cellules ont été réservées : plage épuisée
        u128 c = perm.apply((u128)idx);
        bool abandoned = false;
        for (size_t i = 0; i < segments.size(); ++i) {
            if (must_stop()) { abandoned = true; break; }
            const Segment &seg = segments[i];
            u128 off = cell_to_off(c, seg);
            if (off >= span) continue; // cellule entièrement hors plage (plage non puissance de 2) : rien à scanner
            u128 base = lo + off;
            Point start_pt = scalar_mul_fixed(u128_to_U256(base));
            uint64_t n = (uint64_t)1 << seg.len;
            int64_t r = scan_batch(seg_wt[i], start_pt, n, sc, g_count, g_ts, must_stop,
                                   [&](int64_t hit, int ti) { handle_hit(base + ((u128)hit << seg.start_bit), ti); });
            if (r == -2) { abandoned = true; break; }
        }
        if (abandoned) { g_nb_abandoned = true; break; }
    }
}

// Retourne true si TOUTES les cellules ont été entièrement scannées (plage épuisée).
static bool run_neighbor_mode(u128 lo, u128 hi, int level, int N, unsigned nt) {
    u128 span = hi - lo + (u128)1;
    // free_bits : nombre de bits nécessaires pour couvrir [0, span). Pour une plage "bits"
    // classique (span = 2^(bits-1)), free_bits == bits - 1 exactement comme avant.
    int free_bits = u128_bitlen(span - (u128)1);
    if (!segment_fits(free_bits, N)) {
        std::cerr << "Erreur : " << N << " bits ne tient pas dans cette plage (" << free_bits
                  << " bit(s) utile(s) au maximum)\n";
        exit(1);
    }
    auto segments = build_segments(level, free_bits, N);
    for (auto &s : segments)
        if (!segment_fits(free_bits, s.len)) {
            std::cerr << "Erreur : le segment '" << s.nom << "' ne tient pas dans la plage.\n";
            exit(1);
        }
    const char *noms[] = {"", "prefixe", "milieu", "suffixe", "prefixe+milieu+suffixe"};
    int k = free_bits - N;                  // bits du motif d'une cellule
    const u128 total_cells = (u128)1 << k;  // nombre de cellules par type de segment
    CellPerm perm;
    {   // clés de la permutation : différentes à chaque lancement (ordre de visite aléatoire)
        std::random_device rd;
        std::mt19937_64 g(((uint64_t)rd() << 32) | rd());
        perm.init(k, g);
    }
    const double total_d = (double)total_cells;

    std::cout << "Plage : " << to_hex(lo) << " -> " << to_hex(hi)
              << " | threads : " << nt << " | mode : voisinage " << noms[level] << " (" << N << " bits)"
              << " | ordre : aléatoire sans doublon (" << fmt_keys(total_d) << " cellules"
              << ", ordre différent à chaque lancement)"
              << " | pas de reprise | implémentation pure (sans bibliothèque)\n";

    // Précalcule la table de fenêtre (WinTable) de chaque segment UNE SEULE FOIS : le point
    // d'incrément d'un segment ne dépend que de son start_bit, fixe pour toute la session.
    std::vector<WinTable> seg_wt(segments.size());
    for (size_t i = 0; i < segments.size(); ++i) {
        Point incPt = scalar_mul_fixed(u128_to_U256((u128)1 << segments[i].start_bit));
        build_wintable(seg_wt[i], incPt);
    }

    g_count = 0;
    g_nb_next = 0;
    g_nb_abandoned = false;
    auto t_start = std::chrono::steady_clock::now();
    std::atomic<bool> fin{false};
    std::thread disp([&] {
        auto t0 = t_start, tl = t0;
        uint64_t last = 0;
        int sleep_ms = std::max(50, (int)(g_display_interval * 1000));
        int step = 0; // 0 : ligne immédiate, 1 : première mesure à 2 s, ensuite : intervalle normal
        while (!fin) {
            int wait_ms = step == 0 ? 0 : (step == 1 ? std::min(2000, sleep_ms) : sleep_ms);
            if (wait_ms > 0) {
            // attente par petits pas : l'arrêt (Ctrl-C, clé trouvée) reste immédiat
            auto tw = std::chrono::steady_clock::now();
            while (!fin && std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count() * 1000 < wait_ms)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            ++step;
            if (fin) break;
            auto now = std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(now - t0).count();
            double dt = std::chrono::duration<double>(now - tl).count();
            uint64_t c = g_count.load();
            double inst = dt > 0 ? (c - last) / dt : 0;
            last = c; tl = now;
            double done = std::min((double)g_nb_next.load(), total_d); // essais lancés (cellules)
            double avg = el > 0 ? done / el : 0;
            {
                std::lock_guard<std::mutex> lp(g_print_mtx);
                printf("\r\033[K%s clés | %s clés/s | essai %s/%s | %.4f%% | écoulé %s | ETA max %s%s",
                       fmt_keys((double)c).c_str(), fmt_keys(inst).c_str(),
                       fmt_keys(done).c_str(), fmt_keys(total_d).c_str(), 100.0 * done / total_d,
                       fmt_time(el).c_str(), fmt_time(avg > 0 ? (total_d - done) / avg : 1e18).c_str(),
                       fmt_targets().c_str());
                fflush(stdout);
            }
        }
    });

    std::vector<std::thread> th;
    th.reserve(nt);
    for (unsigned t = 0; t < nt; ++t)
        th.emplace_back(neighbor_worker, std::cref(perm), total_cells, lo, span, std::cref(segments), std::cref(seg_wt));
    for (auto &t : th) t.join();

    fin = true; disp.join();
    // Plage épuisée = toutes les cellules réservées ET aucun thread n'a abandonné un essai en cours.
    bool exhausted = (u128)g_nb_next.load() >= total_cells && !g_nb_abandoned.load();
    if (exhausted) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        std::lock_guard<std::mutex> lp(g_print_mtx);
        printf("\r\033[K%s clés (%llu) | 100.0000%% | écoulé %s%s",
               fmt_keys((double)g_count.load()).c_str(), (unsigned long long)g_count.load(),
               fmt_time(el).c_str(), fmt_targets().c_str());
    }
    printf("\n");
    return exhausted;
}

static bool self_test() {
    bool ok = true;
    auto check = [&](bool c, const char *name) { if (!c) { fprintf(stderr, "AUTO-TEST ECHEC : %s\n", name); ok = false; } };
    U256 ex;
    parse_hex("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5", ex);
    Point g2 = jac_to_aff(jac_mul(U256{{2,0,0,0}}, G));
    check(eq(g2.x, ex), "2G");
    parse_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", ex);
    check(eq(G.x, ex), "G.x");
    // hash160(G) connu (adresse du puzzle 1 : 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH)
    unsigned char wb[20] = {0x75,0x1e,0x76,0xe8,0x19,0x91,0x96,0xd4,0x54,0x94,0x1c,0x45,0xd1,0xb3,0xa3,0x23,0xf1,0x43,0x3b,0xd6};
    unsigned char comp[33], h[20];
    point_to_compressed(G, comp);
    hash160_bytes(comp, 33, h);
    check(memcmp(h, wb, 20) == 0, "hash160(G)");
    check(eq(pack20(wb), pack20(h)), "pack20");
    // scalar_mul_fixed vs jac_mul sur plusieurs valeurs
    std::mt19937_64 rng(424242);
    for (int i = 0; i < 8; i++) {
        u128 k = ((u128)rng() << 64) | rng();
        k &= (((u128)1 << 80) - 1); if (k == 0) k = 1;
        Point a = scalar_mul_fixed(u128_to_U256(k));
        Point b = jac_to_aff(jac_mul(u128_to_U256(k), G));
        if (!(eq(a.x,b.x) && eq(a.y,b.y))) { check(false, "scalar_mul_fixed"); break; }
    }
    // table de fenêtre (HALF) vs jac_mul
    build_wintable(g_wt_main, G);
    for (int i : {1, 2, 255, 256, 511, 512}) {
        Point j = jac_to_aff(jac_mul(U256{{(u64)i,0,0,0}}, G));
        if (!(eq(j.x, g_wt_main.gx[i]) && eq(j.y, g_wt_main.gy[i]))) { check(false, "table de fenetre"); break; }
    }
    // hash160 vectorisé vs hash160_bytes
    {
        U256 xs[VLANES], hs[VLANES]; u64 ps[VLANES]; int bad = 0;
        for (int i = 0; i < VLANES; i++) {
            u128 k = ((u128)rng() << 64) | rng(); if ((k & (((u128)1<<80)-1)) == 0) k = i + 1;
            k &= (((u128)1 << 80) - 1); if (k == 0) k = i + 1;
            Point p = jac_to_aff(jac_mul(u128_to_U256(k), G));
            xs[i] = p.x; ps[i] = p.y.d[0] & 1;
        }
        hash160_vec(xs, ps, hs);
        for (int i = 0; i < VLANES; i++) {
            unsigned char cc[33], hh[20];
            cc[0] = ps[i] ? 0x03 : 0x02;
            x_to_bytes(xs[i], cc + 1);
            hash160_bytes(cc, 33, hh);
            if (!eq(hs[i], pack20(hh))) bad++;
        }
        check(bad == 0, "hash160_vec vs hash160_bytes");
    }
    // filtre/ensemble de cibles + scan complet multi-cibles : clés 12345 et 9000 dans la fenêtre
    // [8192, 16383], clé 20000 volontairement hors fenêtre (ne doit PAS être signalée).
    {
        TargetSet ts;
        const int tk[3] = {12345, 9000, 20000};
        for (int i = 0; i < 3; i++) {
            Point pk = jac_to_aff(jac_mul(U256{{(u64)tk[i],0,0,0}}, G));
            unsigned char tgt[20];
            point_to_compressed(pk, comp);
            hash160_bytes(comp, 33, tgt);
            ts.add("test", tgt, 0);
        }
        check(ts.find(ts.h[1]) == 1 && ts.find(ZERO) == -1, "TargetSet::find");
        Scratch sc;
        std::atomic<uint64_t> tc{0}; // compteur local : ne pollue pas g_count
        std::vector<std::pair<int64_t,int>> hits;
        Point start = scalar_mul_fixed(u128_to_U256(8192));
        scan_batch(g_wt_main, start, 8192, sc, tc, ts, [](){ return false; },
                   [&](int64_t h, int ti) { hits.push_back({h, ti}); });
        std::sort(hits.begin(), hits.end());
        check(hits.size() == 2 && hits[0] == std::make_pair((int64_t)(9000 - 8192), 1)
                               && hits[1] == std::make_pair((int64_t)(12345 - 8192), 0),
              "scan_batch multi-cibles");
        check(tc.load() == 8192, "scan_batch compteur");
    }
    return ok;
}
#else // __CUDACC__
__device__ __forceinline__ void d_mul64(u64 a, u64 b, u64 &lo, u64 &hi) {
    lo = a * b;
    hi = __umul64hi(a, b);
}
// a*b + c0 + c1, resultat 128 bits (lo,hi). Valide tant que a*b+c0+c1 < 2^128 (toujours
// vrai dans cet engin : voir le commentaire au-dessus de fe_reduce/fe_mul plus bas).
__device__ __forceinline__ void d_mac(u64 &lo_out, u64 &hi_out, u64 a, u64 b, u64 c0, u64 c1) {
    u64 lo, hi; d_mul64(a, b, lo, hi);
    u64 s = lo + c0; u64 c = (s < lo) ? 1ULL : 0ULL;
    hi += c;
    u64 s2 = s + c1; u64 c2 = (s2 < s) ? 1ULL : 0ULL;
    hi += c2;
    lo_out = s2; hi_out = hi;
}
__device__ __forceinline__ u64 d_addc(u64 a, u64 b, u64 cin, u64 &cout) {
    u64 s = a + b; u64 c1 = (s < a) ? 1ULL : 0ULL;
    u64 s2 = s + cin; u64 c2 = (s2 < s) ? 1ULL : 0ULL;
    cout = c1 + c2; return s2;
}
__device__ __forceinline__ u64 d_subb(u64 a, u64 b, u64 bin, u64 &bout) {
    u64 t = a - b; u64 b1 = (a < b) ? 1ULL : 0ULL;
    u64 t2 = t - bin; u64 b2 = (t < bin) ? 1ULL : 0ULL;
    bout = b1 + b2; return t2;
}
__device__ __forceinline__ u64 d_add_raw(U256 &r, const U256 &a, const U256 &b) {
    u64 c = 0;
    r.d[0] = d_addc(a.d[0], b.d[0], c, c);
    r.d[1] = d_addc(a.d[1], b.d[1], c, c);
    r.d[2] = d_addc(a.d[2], b.d[2], c, c);
    r.d[3] = d_addc(a.d[3], b.d[3], c, c);
    return c;
}
__device__ __forceinline__ u64 d_sub_raw(U256 &r, const U256 &a, const U256 &b) {
    u64 c = 0;
    r.d[0] = d_subb(a.d[0], b.d[0], c, c);
    r.d[1] = d_subb(a.d[1], b.d[1], c, c);
    r.d[2] = d_subb(a.d[2], b.d[2], c, c);
    r.d[3] = d_subb(a.d[3], b.d[3], c, c);
    return c;
}
__device__ __forceinline__ bool d_is_zero(const U256 &a) { return (a.d[0]|a.d[1]|a.d[2]|a.d[3]) == 0; }
__device__ __forceinline__ bool d_eq(const U256 &a, const U256 &b) {
    return ((a.d[0]^b.d[0])|(a.d[1]^b.d[1])|(a.d[2]^b.d[2])|(a.d[3]^b.d[3])) == 0;
}

__device__ __constant__ u64 D_P[4] = {0xFFFFFFFEFFFFFC2FULL, ~0ULL, ~0ULL, ~0ULL};
__device__ __constant__ u64 D_KC = 0x1000003D1ULL;

__device__ void d_fe_add(U256 &r, const U256 &a, const U256 &b) {
    U256 t; u64 c = d_add_raw(t, a, b);
    if (c) { U256 k = {{D_KC,0,0,0}}; d_add_raw(t, t, k); }
    bool geq = true;
    for (int i = 3; i >= 0 && geq; --i) if (t.d[i] != D_P[i]) { geq = t.d[i] > D_P[i]; break; }
    if (geq) { U256 pp = {{D_P[0],D_P[1],D_P[2],D_P[3]}}; d_sub_raw(t, t, pp); }
    r = t;
}
__device__ void d_fe_sub(U256 &r, const U256 &a, const U256 &b) {
    U256 t; u64 br = d_sub_raw(t, a, b);
    if (br) { U256 k = {{D_KC,0,0,0}}; d_sub_raw(t, t, k); }
    r = t;
}
__device__ void d_fe_reduce(U256 &r, const u64 *t) {
    u64 r5[5]; u64 carry = 0;
    for (int i = 0; i < 4; i++) {
        u64 lo, hi; d_mac(lo, hi, t[4+i], D_KC, t[i], carry);
        r5[i] = lo; carry = hi;
    }
    r5[4] = carry;
    U256 res; u64 lo0, hi0;
    d_mac(lo0, hi0, r5[4], D_KC, r5[0], 0ULL);
    res.d[0] = lo0; carry = hi0;
    for (int i = 1; i < 4; i++) {
        u64 s = r5[i] + carry; u64 c2 = (s < r5[i]) ? 1ULL : 0ULL;
        res.d[i] = s; carry = c2;
    }
    if (carry) {
        u64 s = res.d[0] + D_KC; u64 c2 = (s < res.d[0]) ? 1ULL : 0ULL;
        res.d[0] = s; carry = c2;
        for (int i = 1; i < 4; i++) {
            u64 s2 = res.d[i] + carry; u64 c3 = (s2 < res.d[i]) ? 1ULL : 0ULL;
            res.d[i] = s2; carry = c3;
        }
    }
    bool geq = true;
    for (int i = 3; i >= 0; --i) if (res.d[i] != D_P[i]) { geq = res.d[i] > D_P[i]; break; }
    if (geq) { U256 pp = {{D_P[0],D_P[1],D_P[2],D_P[3]}}; d_sub_raw(res, res, pp); }
    r = res;
}
__device__ void d_fe_mul(U256 &r, const U256 &a, const U256 &b) {
    u64 t[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++) {
            u64 lo, hi; d_mac(lo, hi, a.d[i], b.d[j], t[i+j], carry);
            t[i+j] = lo; carry = hi;
        }
        t[i+4] = carry;
    }
    d_fe_reduce(r, t);
}
__device__ __forceinline__ void d_fe_sqr(U256 &r, const U256 &a) { d_fe_mul(r, a, a); }
__device__ __forceinline__ void d_fe_sqr_n(U256 &r, const U256 &a, int n) {
    U256 t = a; for (int i = 0; i < n; i++) d_fe_sqr(t, t); r = t;
}
__device__ U256 d_fe_inv(const U256 &a) {
    U256 x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t;
    d_fe_sqr(x2,a); d_fe_mul(x2,x2,a);
    d_fe_sqr(x3,x2); d_fe_mul(x3,x3,a);
    d_fe_sqr_n(x6,x3,3); d_fe_mul(x6,x6,x3);
    d_fe_sqr_n(x9,x6,3); d_fe_mul(x9,x9,x3);
    d_fe_sqr_n(x11,x9,2); d_fe_mul(x11,x11,x2);
    d_fe_sqr_n(x22,x11,11); d_fe_mul(x22,x22,x11);
    d_fe_sqr_n(x44,x22,22); d_fe_mul(x44,x44,x22);
    d_fe_sqr_n(x88,x44,44); d_fe_mul(x88,x88,x44);
    d_fe_sqr_n(x176,x88,88); d_fe_mul(x176,x176,x88);
    d_fe_sqr_n(x220,x176,44); d_fe_mul(x220,x220,x44);
    d_fe_sqr_n(x223,x220,3); d_fe_mul(x223,x223,x3);
    d_fe_sqr_n(t,x223,23); d_fe_mul(t,t,x22);
    d_fe_sqr_n(t,t,5); d_fe_mul(t,t,a);
    d_fe_sqr_n(t,t,3); d_fe_mul(t,t,x2);
    d_fe_sqr_n(t,t,2); d_fe_mul(t,t,a);
    return t;
}

// ---- courbe, cote device : Jacobien sans point-a-l'infini explicite (jamais rencontre
// en pratique ici : G et ses petits multiples ne sont jamais le point neutre) ----
struct DJac { U256 X, Y, Z; };
__device__ DJac d_jac_dbl(const DJac &P) {
    U256 A,B,C,D,E,F,t,t2;
    d_fe_sqr(A,P.X); d_fe_sqr(B,P.Y); d_fe_sqr(C,B);
    d_fe_add(t,P.X,B); d_fe_sqr(t,t); d_fe_sub(t,t,A); d_fe_sub(t,t,C); d_fe_add(D,t,t);
    d_fe_add(E,A,A); d_fe_add(E,E,A); d_fe_sqr(F,E);
    DJac R;
    d_fe_add(t,D,D); d_fe_sub(R.X,F,t);
    d_fe_sub(t,D,R.X); d_fe_mul(t,E,t);
    d_fe_add(t2,C,C); d_fe_add(t2,t2,t2); d_fe_add(t2,t2,t2);
    d_fe_sub(R.Y,t,t2);
    d_fe_mul(t,P.Y,P.Z); d_fe_add(R.Z,t,t);
    return R;
}
// addition mixte Jacobien + affine (Q = (Qx,Qy)), copie de jac_madd (hote), sans le cas
// "P.inf" (P est toujours un point concret ici).
__device__ DJac d_jac_madd(const DJac &P, const U256 &Qx, const U256 &Qy) {
    U256 Z1Z1,U2,S2,H,HH,I,J,r,V,t;
    d_fe_sqr(Z1Z1,P.Z); d_fe_mul(U2,Qx,Z1Z1);
    d_fe_mul(S2,Qy,P.Z); d_fe_mul(S2,S2,Z1Z1);
    d_fe_sub(H,U2,P.X);
    d_fe_sub(t,S2,P.Y); d_fe_add(r,t,t);
    if (d_is_zero(H)) {
        if (d_is_zero(r)) return d_jac_dbl(P);
        return P; // Q == -P (probabilite nulle en pratique) : repli inoffensif
    }
    d_fe_sqr(HH,H); d_fe_add(I,HH,HH); d_fe_add(I,I,I);
    d_fe_mul(J,H,I); d_fe_mul(V,P.X,I);
    DJac R;
    d_fe_sqr(R.X,r); d_fe_sub(R.X,R.X,J); d_fe_sub(R.X,R.X,V); d_fe_sub(R.X,R.X,V);
    d_fe_sub(t,V,R.X); d_fe_mul(t,r,t);
    U256 u; d_fe_mul(u,P.Y,J); d_fe_add(u,u,u);
    d_fe_sub(R.Y,t,u);
    d_fe_add(t,P.Z,H); d_fe_sqr(t,t); d_fe_sub(t,t,Z1Z1); d_fe_sub(R.Z,t,HH);
    return R;
}
// point = start + offset*G, via la meme table a fenetres fixes de 8 bits que le hote
// (gtx/gty = les 8 premieres fenetres de g_gtable, soit 64 bits : largement suffisant
// pour un offset de lot, meme pour un lot de plusieurs milliards de cles).
__device__ DJac d_point_at_offset(const DJac &start, const U256* __restrict__ gtx,
                                   const U256* __restrict__ gty, int gt_size, uint64_t offset) {
    DJac acc = start;
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        unsigned dgt = (unsigned)((offset >> (w*8)) & 0xFF);
        if (dgt) acc = d_jac_madd(acc, gtx[w*gt_size + dgt], gty[w*gt_size + dgt]);
    }
    return acc;
}

// ---- SHA-256 / RIPEMD-160 device : portage DIRECT des fonctions scalaires de puzzle_pur.cpp ----
__device__ __constant__ u32 D_SHA_K[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
__device__ __constant__ unsigned char D_RMD_R1[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2, 4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
__device__ __constant__ unsigned char D_RMD_R2[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14, 12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
__device__ __constant__ unsigned char D_RMD_S1[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12, 9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
__device__ __constant__ unsigned char D_RMD_S2[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8, 8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
__device__ __constant__ u32 D_RMD_K1[5]={0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
__device__ __constant__ u32 D_RMD_K2[5]={0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};

__device__ __forceinline__ u32 d_rotr32(u32 x,int n){ return (x>>n)|(x<<(32-n)); }
__device__ __forceinline__ u32 d_rotl32(u32 x,int n){ return (x<<n)|(x>>(32-n)); }
template<int F> __device__ __forceinline__ u32 d_rmd_f(u32 x,u32 y,u32 z){
    if(F==0) return x^y^z;
    if(F==1) return (x&y)|(~x&z);
    if(F==2) return (x|~y)^z;
    if(F==3) return (x&z)|(y&~z);
    return x^(y|~z);
}
template<int F> __device__ __forceinline__ void d_rmd_seg(const u32* X, u32&a,u32&b,u32&c,u32&d,u32&e,
                                                            u32&a2,u32&b2,u32&c2,u32&d2,u32&e2) {
    #pragma unroll
    for (int j = 16*F; j < 16*F+16; j++) {
        u32 t = d_rotl32(a + d_rmd_f<F>(b,c,d) + X[D_RMD_R1[j]] + D_RMD_K1[F], D_RMD_S1[j]) + e;
        a=e; e=d; d=d_rotl32(c,10); c=b; b=t;
        u32 t2 = d_rotl32(a2 + d_rmd_f<4-F>(b2,c2,d2) + X[D_RMD_R2[j]] + D_RMD_K2[F], D_RMD_S2[j]) + e2;
        a2=e2; e2=d2; d2=d_rotl32(c2,10); c2=b2; b2=t2;
    }
}
// hash160 d'une cle publique compressee (33 octets : parite + X big-endian), directement a
// partir de x (U256) et du bit de parite -> empaquetage identique a pack20() (hote).
__device__ U256 d_hash160_x(const U256 &x, unsigned parity) {
    u32 X0=(u32)(x.d[3]>>32), X1=(u32)x.d[3], X2=(u32)(x.d[2]>>32), X3=(u32)x.d[2];
    u32 X4=(u32)(x.d[1]>>32), X5=(u32)x.d[1], X6=(u32)(x.d[0]>>32), X7=(u32)x.d[0];
    u32 P = 0x02u + parity;
    u32 w[64];
    w[0]=(P<<24)|(X0>>8);
    w[1]=(X0<<24)|(X1>>8); w[2]=(X1<<24)|(X2>>8); w[3]=(X2<<24)|(X3>>8);
    w[4]=(X3<<24)|(X4>>8); w[5]=(X4<<24)|(X5>>8); w[6]=(X5<<24)|(X6>>8); w[7]=(X6<<24)|(X7>>8);
    w[8]=(X7<<24)|0x00800000u;
    for (int i=9;i<15;i++) w[i]=0;
    w[15]=264;
    #pragma unroll
    for (int i=16;i<64;i++) {
        u32 s0=d_rotr32(w[i-15],7)^d_rotr32(w[i-15],18)^(w[i-15]>>3);
        u32 s1=d_rotr32(w[i-2],17)^d_rotr32(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    u32 a=0x6a09e667u,b=0xbb67ae85u,c=0x3c6ef372u,d=0xa54ff53au,e=0x510e527fu,f=0x9b05688cu,g=0x1f83d9abu,h=0x5be0cd19u;
    #pragma unroll
    for (int i=0;i<64;i++) {
        u32 S1=d_rotr32(e,6)^d_rotr32(e,11)^d_rotr32(e,25), ch=(e&f)^(~e&g);
        u32 t1=h+S1+ch+D_SHA_K[i]+w[i];
        u32 S0=d_rotr32(a,2)^d_rotr32(a,13)^d_rotr32(a,22), mj=(a&b)^(a&c)^(b&c);
        u32 t2=S0+mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    u32 H0=a+0x6a09e667u, H1=b+0xbb67ae85u, H2=c+0x3c6ef372u, H3=d+0xa54ff53au;
    u32 H4=e+0x510e527fu, H5=f+0x9b05688cu, H6=g+0x1f83d9abu, H7=h+0x5be0cd19u;
    u32 rw[16];
    rw[0]=(H0>>24)|((H0>>8)&0x0000FF00u)|((H0<<8)&0x00FF0000u)|(H0<<24);
    rw[1]=(H1>>24)|((H1>>8)&0x0000FF00u)|((H1<<8)&0x00FF0000u)|(H1<<24);
    rw[2]=(H2>>24)|((H2>>8)&0x0000FF00u)|((H2<<8)&0x00FF0000u)|(H2<<24);
    rw[3]=(H3>>24)|((H3>>8)&0x0000FF00u)|((H3<<8)&0x00FF0000u)|(H3<<24);
    rw[4]=(H4>>24)|((H4>>8)&0x0000FF00u)|((H4<<8)&0x00FF0000u)|(H4<<24);
    rw[5]=(H5>>24)|((H5>>8)&0x0000FF00u)|((H5<<8)&0x00FF0000u)|(H5<<24);
    rw[6]=(H6>>24)|((H6>>8)&0x0000FF00u)|((H6<<8)&0x00FF0000u)|(H6<<24);
    rw[7]=(H7>>24)|((H7>>8)&0x0000FF00u)|((H7<<8)&0x00FF0000u)|(H7<<24);
    rw[8]=0x80u; for(int i=9;i<14;i++) rw[i]=0; rw[14]=256u; rw[15]=0u;
    u32 ra=0x67452301u, rb=0xEFCDAB89u, rc=0x98BADCFEu, rd=0x10325476u, re=0xC3D2E1F0u;
    u32 ra2=ra, rb2=rb, rc2=rc, rd2=rd, re2=re;
    d_rmd_seg<0>(rw,ra,rb,rc,rd,re,ra2,rb2,rc2,rd2,re2);
    d_rmd_seg<1>(rw,ra,rb,rc,rd,re,ra2,rb2,rc2,rd2,re2);
    d_rmd_seg<2>(rw,ra,rb,rc,rd,re,ra2,rb2,rc2,rd2,re2);
    d_rmd_seg<3>(rw,ra,rb,rc,rd,re,ra2,rb2,rc2,rd2,re2);
    d_rmd_seg<4>(rw,ra,rb,rc,rd,re,ra2,rb2,rc2,rd2,re2);
    u32 o0=0xEFCDAB89u+rc+rd2, o1=0x98BADCFEu+rd+re2, o2=0x10325476u+re+ra2, o3=0xC3D2E1F0u+ra+rb2, o4=0x67452301u+rb+rc2;
    U256 out; out.d[0]=(u64)o0|((u64)o1<<32); out.d[1]=(u64)o2|((u64)o3<<32); out.d[2]=(u64)o4; out.d[3]=0;
    return out;
}

struct HitRec { uint64_t offset; int target_idx; };

// Kernel principal : un thread = une cle (start + tid*G), boucle en "grid-stride" pour
// couvrir tout le lot 'n' quelle que soit la taille de grille lancee.
__global__ void scan_kernel(DJac start, const U256* __restrict__ gtx, const U256* __restrict__ gty,
                             int gt_size, uint64_t n,
                             const U256* __restrict__ targets, int n_targets,
                             const u64* __restrict__ bloom,
                             HitRec* __restrict__ hits, int* hit_count, int max_hits) {
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += stride) {
        DJac P = d_point_at_offset(start, gtx, gty, gt_size, tid);
        U256 zi = d_fe_inv(P.Z), zi2, zi3, ax, ay;
        d_fe_sqr(zi2, zi); d_fe_mul(zi3, zi2, zi);
        d_fe_mul(ax, P.X, zi2); d_fe_mul(ay, P.Y, zi3);
        unsigned parity = (unsigned)(ay.d[0] & 1ULL);
        U256 hp = d_hash160_x(ax, parity);
        u64 ix = hp.d[0] & 0xFFFFULL;
        if ((bloom[ix >> 6] >> (ix & 63)) & 1ULL) {
            for (int i = 0; i < n_targets; i++) {
                if (d_eq(hp, targets[i])) {
                    int slot = atomicAdd(hit_count, 1);
                    if (slot < max_hits) { hits[slot].offset = tid; hits[slot].target_idx = i; }
                    break;
                }
            }
        }
    }
}

// Kernel du mode voisinage : regroupe PLUSIEURS cellules dans un seul lancement (une cellule
// isolee, souvent quelques dizaines a quelques millions de cles, sous-utiliserait le GPU). Un
// indice plat tid est decoupe en (numero de cellule, position dans la cellule) via un decalage
// de bits (keys_per_cell = 2^len_shift est toujours une puissance de deux). 'gtx/gty' est la
// table du PAS du segment (2^start_bit * G), pas celle de G.
__global__ void scan_kernel_cells(const DJac* __restrict__ starts, int n_cells,
                                   uint64_t keys_per_cell, int len_shift,
                                   const U256* __restrict__ gtx, const U256* __restrict__ gty, int gt_size,
                                   const U256* __restrict__ targets, int n_targets,
                                   const u64* __restrict__ bloom,
                                   HitRec* __restrict__ hits, int* hit_count, int max_hits) {
    uint64_t n = (uint64_t)n_cells * keys_per_cell;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += stride) {
        int cell = (int)(tid >> len_shift);
        uint64_t off = tid & (keys_per_cell - 1);
        DJac P = d_point_at_offset(starts[cell], gtx, gty, gt_size, off);
        U256 zi = d_fe_inv(P.Z), zi2, zi3, ax, ay;
        d_fe_sqr(zi2, zi); d_fe_mul(zi3, zi2, zi);
        d_fe_mul(ax, P.X, zi2); d_fe_mul(ay, P.Y, zi3);
        unsigned parity = (unsigned)(ay.d[0] & 1ULL);
        U256 hp = d_hash160_x(ax, parity);
        u64 ix = hp.d[0] & 0xFFFFULL;
        if ((bloom[ix >> 6] >> (ix & 63)) & 1ULL) {
            for (int i = 0; i < n_targets; i++) {
                if (d_eq(hp, targets[i])) {
                    int slot = atomicAdd(hit_count, 1);
                    if (slot < max_hits) { hits[slot].offset = tid; hits[slot].target_idx = i; } // tid encode (cellule,position)
                    break;
                }
            }
        }
    }
}

// =================== Pilotage HOTE d'un GPU (un thread C++ par carte) ===================
struct GpuCtx {
    U256 *d_gx = nullptr, *d_gy = nullptr;
    U256 *d_targets = nullptr;
    u64 *d_bloom = nullptr;
    HitRec *d_hits = nullptr;
    int *d_hit_count = nullptr;
    static const int MAX_HITS = 1024;
};

static void gpu_upload_tables(GpuCtx &ctx) {
    // Seules les 8 premieres fenetres (64 bits) de g_gtable sont necessaires pour un offset
    // de lot (uint64_t) ; on les copie dans un tableau plat.
    const int windows = 8;
    std::vector<U256> hx((size_t)windows*GT_SIZE), hy((size_t)windows*GT_SIZE);
    for (int w = 0; w < windows; w++)
        for (int d = 0; d < GT_SIZE; d++) {
            hx[(size_t)w*GT_SIZE+d] = g_gtable[(size_t)w*GT_SIZE+d].x;
            hy[(size_t)w*GT_SIZE+d] = g_gtable[(size_t)w*GT_SIZE+d].y;
        }
    CUDA_CHECK(cudaMalloc(&ctx.d_gx, hx.size()*sizeof(U256)));
    CUDA_CHECK(cudaMalloc(&ctx.d_gy, hy.size()*sizeof(U256)));
    CUDA_CHECK(cudaMemcpy(ctx.d_gx, hx.data(), hx.size()*sizeof(U256), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_gy, hy.data(), hy.size()*sizeof(U256), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&ctx.d_targets, g_ts.h.size()*sizeof(U256)));
    CUDA_CHECK(cudaMemcpy(ctx.d_targets, g_ts.h.data(), g_ts.h.size()*sizeof(U256), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&ctx.d_bloom, sizeof(g_ts.filter)));
    CUDA_CHECK(cudaMemcpy(ctx.d_bloom, g_ts.filter, sizeof(g_ts.filter), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&ctx.d_hits, GpuCtx::MAX_HITS*sizeof(HitRec)));
    CUDA_CHECK(cudaMalloc(&ctx.d_hit_count, sizeof(int)));
}

// Lance le kernel sur [chunk_start, chunk_start+n) et traite les correspondances trouvees.
static void gpu_run_batch(GpuCtx &ctx, u128 chunk_start, uint64_t n) {
    Point sp = scalar_mul_fixed(u128_to_U256(chunk_start));
    DJac start{sp.x, sp.y, ONE};
    CUDA_CHECK(cudaMemset(ctx.d_hit_count, 0, sizeof(int)));
    const int block = 256;
    int grid = (int)std::min<uint64_t>(65535ULL, (n + block - 1) / block);
    if (grid < 1) grid = 1;
    scan_kernel<<<grid, block>>>(start, ctx.d_gx, ctx.d_gy, GT_SIZE, n,
                                  ctx.d_targets, (int)g_ts.size(), ctx.d_bloom,
                                  ctx.d_hits, ctx.d_hit_count, GpuCtx::MAX_HITS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int hit_count = 0;
    CUDA_CHECK(cudaMemcpy(&hit_count, ctx.d_hit_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (hit_count > 0) {
        int m = std::min(hit_count, GpuCtx::MAX_HITS);
        std::vector<HitRec> hh(m);
        CUDA_CHECK(cudaMemcpy(hh.data(), ctx.d_hits, m*sizeof(HitRec), cudaMemcpyDeviceToHost));
        for (auto &hrec : hh) handle_hit(chunk_start + (u128)hrec.offset, hrec.target_idx);
    }
}

// Auto-test GPU : verifie que scan_kernel retrouve bien une cle connue (offset 4 depuis G,
// soit la cle 5) avant de lancer un vrai balayage. Indispensable puisque ce fichier n'a pas
// pu etre compile/execute sur un vrai GPU avant Kaggle.
static bool gpu_self_test(GpuCtx &ctx) {
    Point p5 = jac_to_aff(jac_mul(U256{{5,0,0,0}}, G));
    unsigned char comp[33], h20[20];
    point_to_compressed(p5, comp);
    hash160_bytes(comp, 33, h20);
    TargetSet ts_save = g_ts; // on bascule temporairement sur une cible de test
    GpuCtx tmp = ctx;
    TargetSet test_ts; test_ts.add("test", h20, 0);
    g_ts = test_ts;
    CUDA_CHECK(cudaMemcpy(tmp.d_targets, g_ts.h.data(), g_ts.h.size()*sizeof(U256), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp.d_bloom, g_ts.filter, sizeof(g_ts.filter), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(tmp.d_hit_count, 0, sizeof(int)));
    DJac start{G.x, G.y, ONE}; // depart = 1*G
    scan_kernel<<<1, 32>>>(start, tmp.d_gx, tmp.d_gy, GT_SIZE, 8,
                            tmp.d_targets, (int)g_ts.size(), tmp.d_bloom,
                            tmp.d_hits, tmp.d_hit_count, GpuCtx::MAX_HITS);
    cudaError_t kerr = cudaGetLastError();
    bool ok = true;
    if (kerr != cudaSuccess) {
        std::cerr << "AUTO-TEST GPU ECHEC (lancement du kernel) : " << cudaGetErrorString(kerr) << "\n";
        ok = false;
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
        int hc = 0; CUDA_CHECK(cudaMemcpy(&hc, tmp.d_hit_count, sizeof(int), cudaMemcpyDeviceToHost));
        HitRec hr{UINT64_MAX, -1};
        if (hc > 0) CUDA_CHECK(cudaMemcpy(&hr, tmp.d_hits, sizeof(HitRec), cudaMemcpyDeviceToHost));
        if (hc != 1 || hr.offset != 4 || hr.target_idx != 0) {
            std::cerr << "AUTO-TEST GPU ECHEC : attendu 1 correspondance a l'offset 4, obtenu "
                      << hc << " correspondance(s)"
                      << (hc > 0 ? (", offset=" + std::to_string((unsigned long long)hr.offset)) : "") << "\n";
            ok = false;
        }
    }
    g_ts = ts_save; // restaure les vraies cibles
    CUDA_CHECK(cudaMemcpy(ctx.d_targets, g_ts.h.data(), g_ts.h.size()*sizeof(U256), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_bloom, g_ts.filter, sizeof(g_ts.filter), cudaMemcpyHostToDevice));
    return ok;
}

// ============================ Mode voisinage, cote GPU ============================
struct GpuCtxNeighbor {
    std::vector<U256*> d_gx, d_gy;      // une table (pas du segment) par segment
    DJac *d_starts = nullptr;
    size_t starts_cap = 0;
    U256 *d_targets = nullptr;
    u64 *d_bloom = nullptr;
    HitRec *d_hits = nullptr;
    int *d_hit_count = nullptr;
    static const int MAX_HITS = 1024;
};
static void gpu_upload_neighbor_tables(GpuCtxNeighbor &ctx, const std::vector<Segment> &segments) {
    ctx.d_gx.assign(segments.size(), nullptr);
    ctx.d_gy.assign(segments.size(), nullptr);
    for (size_t i = 0; i < segments.size(); ++i) {
        Point incPt = scalar_mul_fixed(u128_to_U256((u128)1 << segments[i].start_bit));
        std::vector<Point> table = build_fixed_table(incPt);
        std::vector<U256> hx(table.size()), hy(table.size());
        for (size_t j = 0; j < table.size(); ++j) { hx[j] = table[j].x; hy[j] = table[j].y; }
        CUDA_CHECK(cudaMalloc(&ctx.d_gx[i], hx.size()*sizeof(U256)));
        CUDA_CHECK(cudaMalloc(&ctx.d_gy[i], hy.size()*sizeof(U256)));
        CUDA_CHECK(cudaMemcpy(ctx.d_gx[i], hx.data(), hx.size()*sizeof(U256), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx.d_gy[i], hy.data(), hy.size()*sizeof(U256), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&ctx.d_targets, g_ts.h.size()*sizeof(U256)));
    CUDA_CHECK(cudaMemcpy(ctx.d_targets, g_ts.h.data(), g_ts.h.size()*sizeof(U256), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&ctx.d_bloom, sizeof(g_ts.filter)));
    CUDA_CHECK(cudaMemcpy(ctx.d_bloom, g_ts.filter, sizeof(g_ts.filter), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&ctx.d_hits, GpuCtxNeighbor::MAX_HITS*sizeof(HitRec)));
    CUDA_CHECK(cudaMalloc(&ctx.d_hit_count, sizeof(int)));
}
static void gpu_ensure_starts_capacity(GpuCtxNeighbor &ctx, size_t need) {
    if (need <= ctx.starts_cap) return;
    if (ctx.d_starts) CUDA_CHECK(cudaFree(ctx.d_starts));
    CUDA_CHECK(cudaMalloc(&ctx.d_starts, need*sizeof(DJac)));
    ctx.starts_cap = need;
}
// Traite jusqu'a 'want' cellules a partir de l'indice global 'start_idx' pour UN segment : calcule
// (cote hote) le point de depart de chaque cellule valide (off < span), les envoie au GPU en un
// seul lot, et reconstitue la cle reelle pour chaque correspondance trouvee.
static void gpu_run_neighbor_batch(GpuCtxNeighbor &ctx, const CellPerm &perm, u128 lo, u128 span,
                                    const Segment &seg, int seg_index, uint64_t start_idx, uint64_t want) {
    std::vector<DJac> starts; std::vector<uint64_t> orig_idx;
    starts.reserve(want); orig_idx.reserve(want);
    for (uint64_t j = 0; j < want; j++) {
        uint64_t idx = start_idx + j;
        u128 c = perm.apply((u128)idx);
        u128 off = cell_to_off(c, seg);
        if (off >= span) continue; // cellule hors plage (plage non puissance de 2) : rien a scanner
        Point sp = scalar_mul_fixed(u128_to_U256(lo + off));
        starts.push_back(DJac{sp.x, sp.y, ONE});
        orig_idx.push_back(idx);
    }
    if (starts.empty()) return;
    gpu_ensure_starts_capacity(ctx, starts.size());
    CUDA_CHECK(cudaMemcpy(ctx.d_starts, starts.data(), starts.size()*sizeof(DJac), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx.d_hit_count, 0, sizeof(int)));
    uint64_t keys_per_cell = (uint64_t)1 << seg.len;
    uint64_t n = (uint64_t)starts.size() * keys_per_cell;
    const int block = 256;
    int grid = (int)std::min<uint64_t>(65535ULL, (n + block - 1) / block);
    if (grid < 1) grid = 1;
    scan_kernel_cells<<<grid, block>>>(ctx.d_starts, (int)starts.size(), keys_per_cell, seg.len,
                                         ctx.d_gx[seg_index], ctx.d_gy[seg_index], GT_SIZE,
                                         ctx.d_targets, (int)g_ts.size(), ctx.d_bloom,
                                         ctx.d_hits, ctx.d_hit_count, GpuCtxNeighbor::MAX_HITS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int hit_count = 0;
    CUDA_CHECK(cudaMemcpy(&hit_count, ctx.d_hit_count, sizeof(int), cudaMemcpyDeviceToHost));
    g_count.fetch_add(n, std::memory_order_relaxed);
    if (hit_count > 0) {
        int m = std::min(hit_count, GpuCtxNeighbor::MAX_HITS);
        std::vector<HitRec> hh(m);
        CUDA_CHECK(cudaMemcpy(hh.data(), ctx.d_hits, m*sizeof(HitRec), cudaMemcpyDeviceToHost));
        for (auto &hrec : hh) {
            int cell_i = (int)(hrec.offset >> seg.len);
            uint64_t off_in_seg = hrec.offset & (keys_per_cell - 1);
            uint64_t idx = orig_idx[cell_i];
            u128 c = perm.apply((u128)idx);
            u128 off = cell_to_off(c, seg);
            u128 base = lo + off;
            handle_hit(base + ((u128)off_in_seg << seg.start_bit), hrec.target_idx);
        }
    }
}
// Auto-test dedie au kernel multi-cellules : verifie le DECODAGE (cellule, position) depuis
// l'indice plat, independamment des vrais segments. 2 cellules factices demarrant tres loin
// l'une de l'autre (1*G et 100*G, par lots de 4 cles -> aucun chevauchement possible entre les
// deux cellules) ; la cle 102 = 100*G + 2*G doit etre UNIQUEMENT trouvee a l'indice plat 1*4+2=6.
static bool gpu_neighbor_self_test() {
    Point p102 = jac_to_aff(jac_mul(U256{{102,0,0,0}}, G));
    unsigned char comp[33], h20[20];
    point_to_compressed(p102, comp);
    hash160_bytes(comp, 33, h20);
    TargetSet test_ts; test_ts.add("test", h20, 0);

    std::vector<U256> hx(8*GT_SIZE), hy(8*GT_SIZE);
    for (int w = 0; w < 8; w++) for (int d = 0; d < GT_SIZE; d++) {
        hx[w*GT_SIZE+d] = g_gtable[w*GT_SIZE+d].x; hy[w*GT_SIZE+d] = g_gtable[w*GT_SIZE+d].y;
    }
    U256 *d_gx, *d_gy;
    CUDA_CHECK(cudaMalloc(&d_gx, hx.size()*sizeof(U256)));
    CUDA_CHECK(cudaMalloc(&d_gy, hy.size()*sizeof(U256)));
    CUDA_CHECK(cudaMemcpy(d_gx, hx.data(), hx.size()*sizeof(U256), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy, hy.data(), hy.size()*sizeof(U256), cudaMemcpyHostToDevice));

    Point g100 = jac_to_aff(jac_mul(U256{{100,0,0,0}}, G));
    DJac starts_h[2] = { DJac{G.x, G.y, ONE}, DJac{g100.x, g100.y, ONE} };
    DJac *d_starts; CUDA_CHECK(cudaMalloc(&d_starts, 2*sizeof(DJac)));
    CUDA_CHECK(cudaMemcpy(d_starts, starts_h, 2*sizeof(DJac), cudaMemcpyHostToDevice));

    U256 *d_targets; CUDA_CHECK(cudaMalloc(&d_targets, test_ts.h.size()*sizeof(U256)));
    CUDA_CHECK(cudaMemcpy(d_targets, test_ts.h.data(), test_ts.h.size()*sizeof(U256), cudaMemcpyHostToDevice));
    u64 *d_bloom; CUDA_CHECK(cudaMalloc(&d_bloom, sizeof(test_ts.filter)));
    CUDA_CHECK(cudaMemcpy(d_bloom, test_ts.filter, sizeof(test_ts.filter), cudaMemcpyHostToDevice));
    HitRec *d_hits; int *d_hit_count;
    CUDA_CHECK(cudaMalloc(&d_hits, 64*sizeof(HitRec)));
    CUDA_CHECK(cudaMalloc(&d_hit_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hit_count, 0, sizeof(int)));

    scan_kernel_cells<<<1,32>>>(d_starts, 2, 4, 2, d_gx, d_gy, GT_SIZE, d_targets, (int)test_ts.size(),
                                 d_bloom, d_hits, d_hit_count, 64);
    cudaError_t kerr = cudaGetLastError();
    bool ok = true;
    if (kerr != cudaSuccess) {
        std::cerr << "AUTO-TEST GPU (voisinage) ECHEC (lancement) : " << cudaGetErrorString(kerr) << "\n";
        ok = false;
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
        int hc = 0; CUDA_CHECK(cudaMemcpy(&hc, d_hit_count, sizeof(int), cudaMemcpyDeviceToHost));
        HitRec hr{UINT64_MAX, -1};
        if (hc > 0) CUDA_CHECK(cudaMemcpy(&hr, d_hits, sizeof(HitRec), cudaMemcpyDeviceToHost));
        if (hc != 1 || hr.offset != 6 || hr.target_idx != 0) {
            std::cerr << "AUTO-TEST GPU (voisinage) ECHEC : attendu 1 correspondance a l'indice plat 6, obtenu "
                      << hc << " correspondance(s)"
                      << (hc > 0 ? (", indice=" + std::to_string((unsigned long long)hr.offset)) : "") << "\n";
            ok = false;
        }
    }
    cudaFree(d_gx); cudaFree(d_gy); cudaFree(d_starts); cudaFree(d_targets);
    cudaFree(d_bloom); cudaFree(d_hits); cudaFree(d_hit_count);
    return ok;
}

static std::atomic<uint64_t> g_nb_next{0};

// Thread de pilotage d'une carte GPU en mode voisinage : reclame des LOTS de cellules (pas une
// cellule a la fois, pour saturer le GPU meme quand chaque cellule est minuscule), pour chaque
// lot lance un kernel par segment, jusqu'a epuisement des cellules ou arret demande.
static void gpu_neighbor_worker(int device_id, const CellPerm &perm, u128 total_cells, u128 lo, u128 span,
                                 const std::vector<Segment> &segments, int target_threads_log2) {
    CUDA_CHECK(cudaSetDevice(device_id));
    if (!gpu_neighbor_self_test()) {
        std::lock_guard<std::mutex> lp(g_print_mtx);
        std::cerr << "GPU #" << device_id << " : auto-test voisinage echoue, cette carte est desactivee.\n";
        return;
    }
    GpuCtxNeighbor ctx;
    gpu_upload_neighbor_tables(ctx, segments);
    {
        std::lock_guard<std::mutex> lp(g_print_mtx);
        std::cout << "GPU #" << device_id << " : auto-test voisinage OK, demarrage.\n";
    }
    // Tous les segments d'un meme essai partagent la meme longueur 'len' (voir build_segments :
    // prefixe/milieu/suffixe ont toujours len == N), donc une seule taille de lot suffit pour tous.
    const int len = segments[0].len;
    const uint64_t target = 1ULL << target_threads_log2;
    const uint64_t MAX_BATCH_CELLS = 1ULL << 16;
    uint64_t want = target >> len; if (want < 1) want = 1;
    if (want > MAX_BATCH_CELLS) want = MAX_BATCH_CELLS;
    while (!g_stop.load(std::memory_order_relaxed) && !g_found.load(std::memory_order_relaxed)) {
        uint64_t start_idx = g_nb_next.fetch_add(want, std::memory_order_relaxed);
        if ((u128)start_idx >= total_cells) break;
        uint64_t actual = (uint64_t)std::min<u128>((u128)want, total_cells - (u128)start_idx);
        for (size_t i = 0; i < segments.size(); ++i)
            gpu_run_neighbor_batch(ctx, perm, lo, span, segments[i], (int)i, start_idx, actual);
    }
}

// Retourne true si TOUTES les cellules ont ete entierement reclamees (plage epuisee) sans arret
// demande entre-temps.
static bool gpu_run_neighbor_mode(u128 lo, u128 hi, int level, int N, int ndev, int target_threads_log2) {
    u128 span = hi - lo + (u128)1;
    int free_bits = u128_bitlen(span - (u128)1);
    if (!segment_fits(free_bits, N)) {
        std::cerr << "Erreur : " << N << " bits ne tient pas dans cette plage (" << free_bits
                  << " bit(s) utile(s) au maximum)\n";
        return false;
    }
    auto segments = build_segments(level, free_bits, N);
    for (auto &s : segments)
        if (!segment_fits(free_bits, s.len)) { std::cerr << "Erreur : le segment '" << s.nom << "' ne tient pas dans la plage.\n"; return false; }
    const char *noms[] = {"", "prefixe", "milieu", "suffixe", "prefixe+milieu+suffixe"};
    int k = free_bits - N;
    const u128 total_cells = (u128)1 << k;
    CellPerm perm;
    { std::random_device rd; std::mt19937_64 g(((uint64_t)rd() << 32) | rd()); perm.init(k, g); }
    const double total_d = (double)total_cells;

    std::cout << "Plage : " << to_hex(lo) << " -> " << to_hex(hi) << " | GPU : " << ndev
              << " | mode : voisinage " << noms[level] << " (" << N << " bits)"
              << " | ordre : aleatoire sans doublon (" << fmt_keys(total_d) << " cellules)\n";

    g_count = 0; g_nb_next = 0;
    std::vector<std::thread> th;
    for (int i = 0; i < ndev; ++i)
        th.emplace_back(gpu_neighbor_worker, i, std::cref(perm), total_cells, lo, span, std::cref(segments), target_threads_log2);

    auto t0 = std::chrono::steady_clock::now(), tl = t0;
    uint64_t last = 0;
    int sleep_ms = std::max(50, (int)(g_display_interval * 1000));
    const int poll_ms = 50; int since_print_ms = sleep_ms;
    auto all_done = [&] { return (u128)g_nb_next.load() >= total_cells; };
    while (!g_stop && !g_found && !all_done()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
        since_print_ms += poll_ms;
        if (g_stop || g_found || all_done()) break;
        if (since_print_ms < sleep_ms) continue;
        since_print_ms = 0;
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - t0).count();
        double dt = std::chrono::duration<double>(now - tl).count();
        uint64_t c = g_count.load();
        double inst = dt > 0 ? (c - last) / dt : 0; last = c; tl = now;
        double done = std::min((double)g_nb_next.load(), total_d);
        double avg = el > 0 ? done / el : 0;
        std::lock_guard<std::mutex> lp(g_print_mtx);
        printf("\r\033[K%s cles | %s cles/s | essai %s/%s | %.4f%% | ecoule %s | ETA max %s%s",
               fmt_keys((double)c).c_str(), fmt_keys(inst).c_str(), fmt_keys(done).c_str(), fmt_keys(total_d).c_str(),
               100.0 * done / total_d, fmt_time(el).c_str(), fmt_time(avg > 0 ? (total_d - done) / avg : 1e18).c_str(),
               fmt_targets().c_str());
        fflush(stdout);
    }
    for (auto &t : th) t.join();
    bool exhausted = !g_stop.load() && (u128)g_nb_next.load() >= total_cells;
    if (exhausted) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::lock_guard<std::mutex> lp(g_print_mtx);
        printf("\r\033[K%s cles (%llu) | 100.0000%% | ecoule %s%s", fmt_keys((double)g_count.load()).c_str(),
               (unsigned long long)g_count.load(), fmt_time(el).c_str(), fmt_targets().c_str());
        fflush(stdout);
    }
    printf("\n");
    return exhausted;
}

static bool host_self_test() {
    bool ok = true;
    auto check = [&](bool c, const char *name) { if (!c) { fprintf(stderr, "AUTO-TEST ECHEC : %s\n", name); ok = false; } };
    U256 ex;
    parse_hex("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5", ex);
    Point g2 = jac_to_aff(jac_mul(U256{{2,0,0,0}}, G));
    check(eq(g2.x, ex), "2G");
    unsigned char wb[20] = {0x75,0x1e,0x76,0xe8,0x19,0x91,0x96,0xd4,0x54,0x94,0x1c,0x45,0xd1,0xb3,0xa3,0x23,0xf1,0x43,0x3b,0xd6};
    unsigned char comp[33], h[20];
    point_to_compressed(G, comp);
    hash160_bytes(comp, 33, h);
    check(memcmp(h, wb, 20) == 0, "hash160(G)");
    std::mt19937_64 rng(424242);
    for (int i = 0; i < 8; i++) {
        u128 k = ((u128)rng() << 64) | rng();
        k &= (((u128)1 << 80) - 1); if (k == 0) k = 1;
        Point a = scalar_mul_fixed(u128_to_U256(k));
        Point b = jac_to_aff(jac_mul(u128_to_U256(k), G));
        if (!(eq(a.x,b.x) && eq(a.y,b.y))) { check(false, "scalar_mul_fixed"); break; }
    }
    return ok;
}

// ---- Thread de pilotage d'une carte GPU : reclame des lots dans la file partagee, lance
// le kernel, traite les correspondances, jusqu'a epuisement de la plage ou arret demande. ----
static void gpu_worker(int device_id, unsigned slot, u128 lo, u128 span, int cb) {
    CUDA_CHECK(cudaSetDevice(device_id));
    GpuCtx ctx; gpu_upload_tables(ctx);
    if (!gpu_self_test(ctx)) {
        std::lock_guard<std::mutex> lp(g_print_mtx);
        std::cerr << "GPU #" << device_id << " : auto-test echoue, cette carte est desactivee.\n";
        return;
    }
    {
        std::lock_guard<std::mutex> lp(g_print_mtx);
        std::cout << "GPU #" << device_id << " : auto-test OK, demarrage du balayage.\n";
    }
    const uint64_t n_full = 1ULL << cb;
    uint64_t idx;
    while (!g_stop.load(std::memory_order_relaxed) && claim(slot, idx)) {
        u128 offset = (u128)idx << cb; // idx < g_nchunks par construction de claim()
        if (offset >= span) { release(slot); continue; }
        uint64_t n = (uint64_t)std::min<u128>((u128)n_full, span - offset);
        gpu_run_batch(ctx, lo + offset, n);
        g_count.fetch_add(n, std::memory_order_relaxed);
        release(slot);
    }
}

#endif // __CUDACC__

// ---- Adresses du Bitcoin Puzzle n° 1 à 80 (index 0 = puzzle 1) ----
// Plage de clés du puzzle P : [2^(P-1), 2^P - 1]
static const char *PUZZLE_ADDR[80] = {
    "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH", "1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb",
    "19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA", "1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e",
    "1E6NuFjCi27W5zoXg8TRdcSRq84zJeBW3k", "1PitScNLyp2HCygzadCh7FveTnfmpPbfp8",
    "1McVt1vMtCC7yn5b9wgX1833yCcLXzueeC", "1M92tSqNmQLYw33fuBvjmeadirh1ysMBxK",
    "1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV", "1LeBZP5QCwwgXRtmVUvTVrraqPUokyLHqe",
    "1PgQVLmst3Z314JrQn5TNiys8Hc38TcXJu", "1DBaumZxUkM4qMQRt2LVWyFJq5kDtSZQot",
    "1Pie8JkxBT6MGPz9Nvi3fsPkr2D8q3GBc1", "1ErZWg5cFCe4Vw5BzgfzB74VNLaXEiEkhk",
    "1QCbW9HWnwQWiQqVo5exhAnmfqKRrCRsvW", "1BDyrQ6WoF8VN3g9SAS1iKZcPzFfnDVieY",
    "1HduPEXZRdG26SUT5Yk83mLkPyjnZuJ7Bm", "1GnNTmTVLZiqQfLbAdp9DVdicEnB5GoERE",
    "1NWmZRpHH4XSPwsW6dsS3nrNWfL1yrJj4w", "1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum",
    "14oFNXucftsHiUMY8uctg6N487riuyXs4h", "1CfZWK1QTQE3eS9qn61dQjV89KDjZzfNcv",
    "1L2GM8eE7mJWLdo3HZS6su1832NX2txaac", "1rSnXMr63jdCuegJFuidJqWxUPV7AtUf7",
    "15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP", "1JVnST957hGztonaWK6FougdtjxzHzRMMg",
    "128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k", "12jbtzBb54r97TCwW3G1gCFoumpckRAPdY",
    "19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT", "1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps",
    "1LhE6sCTuGae42Axu1L1ZB7L96yi9irEBE", "1FRoHA9xewq7DjrZ1psWJVeTer8gHRqEvR",
    "187swFMjz1G54ycVU56B7jZFHFTNVQFDiu", "1PWABE7oUahG2AFFQhhvViQovnCr4rEv7Q",
    "1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb", "1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1",
    "14iXhn8bGajVWegZHJ18vJLHhntcpL4dex", "1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2",
    "122AJhKLEfkFBaGAd84pLp1kfE7xK3GdT8", "1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv",
    "1L5sU9qvJeuwQUdt4y1eiLmquFxKjtHr3E", "1E32GPWgDyeyQac4aJxm9HVoLrrEYPnM4N",
    "1PiFuqGpG8yGM5v6rNHWS3TjsG6awgEGA1", "1CkR2uS7LmFwc3T2jV8C1BhWb5mQaoxedF",
    "1NtiLNGegHWE3Mp9g2JPkgx6wUg4TW7bbk", "1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP",
    "1Pd8VvT49sHKsmqrQiP61RsVwmXCZ6ay7Z", "1DFYhaB2J9q1LLZJWKTnscPWos9VBqDHzv",
    "12CiUhYVTTH33w3SPUBqcpMoqnApAV4WCF", "1MEzite4ReNuWaL5Ds17ePKt2dCxWEofwk",
    "1NpnQyZ7x24ud82b7WiRNvPm6N8bqGQnaS", "15z9c9sVpu6fwNiK7dMAFgMYSK4GqsGZim",
    "15K1YKJMiJ4fpesTVUcByoz334rHmknxmT", "1KYUv7nSvXx4642TKeuC2SNdTk326uUpFy",
    "1LzhS3k3e9Ub8i2W1V8xQFdB8n2MYCHPCa", "17aPYR1m6pVAacXg1PTDDU7XafvK1dxvhi",
    "15c9mPGLku1HuW9LRtBf4jcHVpBUt8txKz", "1Dn8NF8qDyyfHMktmuoQLGyjWmZXgvosXf",
    "1HAX2n9Uruu9YDt4cqRgYcvtGvZj1rbUyt", "1Kn5h2qpgw9mWE5jKpk8PP4qvvJ1QVy8su",
    "1AVJKwzs9AskraJLGHAZPiaZcrpDr1U6AB", "1Me6EfpwZK5kQziBwBfvLiHjaPGxCKLoJi",
    "1NpYjtLira16LfGbGwZJ5JbDPh3ai9bjf4", "16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN",
    "18ZMbwUFLMHoZBbfpCjUJQTCMCbktshgpe", "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so",
    "1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9", "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ",
    "19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG", "19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR",
    "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU", "1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR",
    "12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4", "1FWGcVDK3JGzCC3WtkYetULPszMaK2Jksv",
    "1J36UjUByGroXcCvmj13U6uwaVv9caEeAt", "1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF",
    "1Bxk4CQdqL9p22JEtDfdXMsng1XacifUtE", "15qF6X51huDjqTmF9BJgxXdt1xcj46Jmhb",
    "1ARk8HWJMn8js8tQmGUJeQHjSE7KRkn2t8", "1BCf6rHUW6m3iH2ptsvnjgLruAiPQQepLe",
};

// Analyse "-pA" ou "-pApB" (A, B = numéros de puzzle). Format strict : -p<chiffres>[p<chiffres>].
static bool parse_puzzle_arg(const std::string &a, int &pa, int &pb) {
    size_t i = 2, j = i;
    while (j < a.size() && isdigit((unsigned char)a[j])) ++j;
    if (j == i || j - i > 3) return false;
    pa = atoi(a.substr(i, j - i).c_str());
    if (j == a.size()) { pb = pa; return true; }
    if (a[j] != 'p') return false;
    size_t k = ++j;
    while (j < a.size() && isdigit((unsigned char)a[j])) ++j;
    if (j != a.size() || j == k || j - k > 3) return false;
    pb = atoi(a.substr(k).c_str());
    return true;
}

static void print_usage(const char *prog) {
    std::cerr << "Usage : " << prog << " -pP | -pApB | <adresse> <bits 1-80 | debut:fin en hex> "
              << "[-vN | -vvN | -vvvN | -vvvvN] [-s T]"
#ifdef __CUDACC__
              << " [-g N]\n"
#else
              << " [-t N] [--seq] [--chunk N]\n"
#endif
              << "  -pP    : puzzle numero P (1-80)\n"
              << "  -pApB  : tous les puzzles de A a B cherches EN MEME TEMPS (ex: -p66p70)\n"
              << "  -vN / -vvN / -vvvN / -vvvvN : mode voisinage (prefixe / milieu / suffixe / les trois), N bits\n"
              << "  -s T   : rafraichit l'affichage toutes les T secondes (defaut 30)\n"
#ifdef __CUDACC__
              << "  -g N   : taille de lot GPU = 2^N cles (ou essais, en voisinage) par lancement de kernel (defaut 23)\n"
              << "Toutes les cartes CUDA visibles sont utilisees automatiquement (ex : les 2 T4 de Kaggle).\n";
#else
              << "  -t N     : nombre de coeurs a utiliser (defaut : tous)\n"
              << "  --seq    : parcourt la plage dans l'ordre sequentiel (defaut : aleatoire sans doublon)\n"
              << "  --chunk N: taille de bloc = 2^N cles par reservation de thread (defaut 10)\n";
#endif
}

static void print_final_summary(bool exhausted, bool lead_nl) {
    int n = (int)g_ts.size(), left = g_remaining.load(), got = n - left;
    if (left == 0) return; // chaque clé a déjà été affichée par save_key
    if (lead_nl) std::cout << "\n";
    const char *why = exhausted ? "Plage entièrement scannée" : "Interrompu";
    if (n == 1) std::cout << why << (exhausted ? " : clé non trouvée.\n" : " (aucune clé trouvée).\n");
    else        std::cout << why << " : " << got << "/" << n << " clé(s) trouvée(s).\n";
}

int main(int argc, char **argv) {
    // ---- 1. Extraction des arguments -pP / -pApB (cumulables) ; le reste est traite plus bas ----
    std::vector<std::string> args(argv + 1, argv + argc), rest;
    bool puzzle_sel[MAX_TARGETS + 1] = {false};
    bool puzzle_mode = false;
    for (const std::string &a : args) {
        if (a.size() >= 3 && a[0] == '-' && a[1] == 'p' && isdigit((unsigned char)a[2])) {
            int pa = 0, pb = 0;
            if (!parse_puzzle_arg(a, pa, pb) || pa < 1 || pa > 80 || pb < 1 || pb > 80) {
                std::cerr << "Argument puzzle invalide : " << a
                          << " (attendu -pP ou -pApB avec 1 <= P <= 80, ex: -p66 ou -p66p70)\n";
                return 1;
            }
            if (pa > pb) std::swap(pa, pb);
            for (int p = pa; p <= pb; ++p) puzzle_sel[p] = true;
            puzzle_mode = true;
        } else rest.push_back(a);
    }

    // ---- 2. Cibles et intervalle ----
    u128 g_lo = 0, g_hi = 0;
    bool custom_range = false;
    int bits = 0;
    size_t opt_start = 0;
    if (puzzle_mode) {
        int pmin = 0, pmax = 0;
        for (int p = 1; p <= 80; ++p) {
            if (!puzzle_sel[p]) continue;
            unsigned char h20[20];
            if (!decode_address(PUZZLE_ADDR[p - 1], h20)) {
                std::cerr << "Erreur interne : adresse du puzzle #" << p << " invalide (checksum)\n";
                return 1;
            }
            g_ts.add(PUZZLE_ADDR[p - 1], h20, p);
            if (!pmin) pmin = p;
            pmax = p;
        }
        bits = pmax;
        g_lo = (u128)1 << (pmin - 1);
        g_hi = ((u128)1 << pmax) - 1;
        if (g_ts.size() == 1) {
            std::cout << "Puzzle #" << pmin << " | adresse : " << g_ts.addr[0] << "\n";
        } else {
            std::cout << "Puzzles #" << pmin << " a #" << pmax << " (" << g_ts.size() << " cibles) | intervalle commun : "
                      << to_hex(g_lo) << " -> " << to_hex(g_hi) << "\n";
            for (size_t i = 0; i < g_ts.size(); ++i)
                std::cout << "  #" << g_ts.puzzle[i] << " : " << g_ts.addr[i] << "\n";
        }
    } else {
        if (rest.size() < 2) { print_usage(argv[0]); return 1; }
        const std::string &addr = rest[0], &arg2 = rest[1];
        opt_start = 2;
        custom_range = arg2.find(':') != std::string::npos;
        if (custom_range) {
            if (!parse_hex_range(arg2, g_lo, g_hi)) {
                std::cerr << "Plage invalide : " << arg2 << " (attendu debut:fin en hexadecimal)\n";
                return 1;
            }
            if (g_lo == g_hi) { std::cerr << "La plage debut:fin doit couvrir au moins 2 cles\n"; return 1; }
            bits = u128_bitlen(g_hi);
        } else {
            bits = atoi(arg2.c_str());
        }
        if (bits < 1 || bits > 80) {
            std::cerr << (custom_range ? "La plage debut:fin doit tenir sur 80 bits au maximum\n"
                                        : "bits doit etre entre 1 et 80\n");
            return 1;
        }
        unsigned char h20[20];
        if (!decode_address(addr, h20)) { std::cerr << "Adresse P2PKH invalide (checksum)\n"; return 1; }
        g_ts.add(addr, h20, 0);
        if (!custom_range) { g_lo = (u128)1 << (bits - 1); g_hi = ((u128)1 << bits) - 1; }
    }
    g_remaining = (int)g_ts.size();
    for (int i = 0; i < MAX_TARGETS; ++i) g_tfound[i] = false;

    // ---- 3. Options : communes (-s, -v) + propres a chaque moteur (l'autre moteur les ignore) ----
    int cpu_chunk_bits = 10;
    int gpu_batch_bits = 23;
    bool seq = false;
    unsigned nt_override = 0;
    int neighbor_level = 0, neighbor_bits = 0;
    for (size_t i = opt_start; i < rest.size(); ++i) {
        const std::string &a = rest[i];
        if (a == "-s" && i + 1 < rest.size()) {
            double v = atof(rest[++i].c_str());
            if (v <= 0) { std::cerr << "-s doit etre suivi d'une duree en secondes > 0\n"; return 1; }
            g_display_interval = v;
        } else if (a.size() >= 2 && a[0] == '-' && a[1] == 'v') {
            size_t p = 2; while (p < a.size() && a[p] == 'v') ++p;
            int level = (int)(p - 1);
            if (level < 1 || level > 4 || p >= a.size() || !isdigit((unsigned char)a[p])) {
                std::cerr << "Argument invalide : " << a << " (attendu -vN, -vvN, -vvvN ou -vvvvN)\n";
                return 1;
            }
            neighbor_level = level; neighbor_bits = atoi(a.c_str() + p);
        }
#ifdef __CUDACC__
        else if (a == "-g" && i + 1 < rest.size()) {
            int v = atoi(rest[++i].c_str());
            if (v < 8 || v > 34) { std::cerr << "-g doit etre entre 8 et 34\n"; return 1; }
            gpu_batch_bits = v;
        }
#else
        else if (a == "--seq") { seq = true; }
        else if (a == "--chunk" && i + 1 < rest.size()) { cpu_chunk_bits = atoi(rest[++i].c_str()); }
        else if (a == "-t" && i + 1 < rest.size()) {
            int v = atoi(rest[++i].c_str());
            if (v < 1) { std::cerr << "-t doit etre suivi d'un nombre de coeurs >= 1\n"; return 1; }
            nt_override = (unsigned)v;
        }
#endif
        else std::cerr << "Avertissement : argument ignore : " << a << "\n";
    }
#ifndef __CUDACC__
    cpu_chunk_bits = std::max(0, std::min(cpu_chunk_bits, bits - 1));
#endif

    U256 gx, gy;
    parse_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", gx);
    parse_hex("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", gy);
    G.x = gx; G.y = gy; G.inf = false;

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    init_gtable(G);

    // ---- 4. Auto-tests et detection des ressources de calcul (different par moteur) ----
#ifdef __CUDACC__
    int ndev = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&ndev);
    if (dev_err != cudaSuccess || ndev == 0) {
        std::cerr << "Aucun GPU CUDA detecte (" << (dev_err == cudaSuccess ? "0 carte" : cudaGetErrorString(dev_err))
                  << "). Recompilez sans \"-x cu\" pour le moteur CPU, ou verifiez l'accelerateur GPU.\n";
        return 1;
    }
    std::cout << ndev << " carte(s) GPU detectee(s) :\n";
    for (int i = 0; i < ndev; i++) {
        cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "  #" << i << " : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    }
    if (!host_self_test()) { std::cerr << "Erreur : auto-test echoue, arret.\n"; return 1; }
#else
    unsigned nt = nt_override ? nt_override : std::max(1u, std::thread::hardware_concurrency());
    auto t_init = std::chrono::steady_clock::now();
    build_wintable(g_wt_main, G);
    if (!self_test()) { std::cerr << "Erreur : auto-test echoue, arret.\n"; return 1; }
    double dt_init = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_init).count();
    std::cout << "Tables precalculees (base fixe + fenetre " << HALF << ") et auto-tests OK en "
              << dt_init << " s | SIMD hash160 : " << VLANES << " voies\n";
#endif

    // ---- 5. Mode voisinage (court-circuite le mode normal ci-dessous) ----
    if (neighbor_level > 0) {
#ifdef __CUDACC__
        bool exhausted = gpu_run_neighbor_mode(g_lo, g_hi, neighbor_level, neighbor_bits, ndev, gpu_batch_bits);
#else
        if (seq || cpu_chunk_bits != 10)
            std::cerr << "Note : --seq et --chunk sont ignores en mode voisinage.\n";
        bool exhausted = run_neighbor_mode(g_lo, g_hi, neighbor_level, neighbor_bits, nt);
#endif
        print_final_summary(exhausted, false);
        return 0;
    }

    // ---- 6. Mode normal (balayage complet de la plage) ----
#ifdef __CUDACC__
    u128 span = g_hi - g_lo + (u128)1;
    {
        int span_bits = u128_bitlen(span - (u128)1);
        if (span_bits - gpu_batch_bits > 62) { gpu_batch_bits = span_bits - 62; std::cerr << "Note : lot GPU porte a 2^" << gpu_batch_bits << ".\n"; }
        u128 chunk_size = (u128)1 << gpu_batch_bits;
        uint64_t nchunks_real = (uint64_t)((span + chunk_size - (u128)1) / chunk_size);
        g_nchunks = next_pow2_u64(nchunks_real == 0 ? 1 : nchunks_real);
    }
    g_cur.assign(ndev, IDLE);

    std::cout << "Plage : " << to_hex(g_lo) << " -> " << to_hex(g_hi) << " | cibles : " << g_ts.size()
              << " | GPU : " << ndev << " | lot : 2^" << gpu_batch_bits << " cles par lancement de kernel\n";

    g_count = 0; g_next = 0;
    std::vector<std::thread> th;
    for (int i = 0; i < ndev; ++i) th.emplace_back(gpu_worker, i, (unsigned)i, g_lo, span, gpu_batch_bits);

    auto t0 = std::chrono::steady_clock::now(), tl = t0;
    uint64_t last = 0;
    const double total = (double)span;
    auto all_idle = [&] {
        std::lock_guard<std::mutex> l(g_mtx);
        if (g_next < g_nchunks) return false;
        for (auto c : g_cur) if (c != IDLE) return false;
        return true;
    };
    int sleep_ms = std::max(50, (int)(g_display_interval * 1000));
    const int poll_ms = 50;
    int since_print_ms = sleep_ms;
    while (!g_stop && !all_idle()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
        since_print_ms += poll_ms;
        if (g_stop || all_idle()) break;
        if (since_print_ms < sleep_ms) continue;
        since_print_ms = 0;
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - t0).count();
        double dt = std::chrono::duration<double>(now - tl).count();
        uint64_t c = g_count.load();
        double inst = (c - last) / dt, avg = c / el;
        last = c; tl = now;
        double remaining = total - (double)c;
        {
            std::lock_guard<std::mutex> lp(g_print_mtx);
            printf("\r\033[K%s cles | %s cles/s | %.4f%% | ecoule %s | ETA max %s%s",
                   fmt_keys((double)c).c_str(), fmt_keys(inst).c_str(), 100.0 * (total - remaining) / total,
                   fmt_time(el).c_str(), fmt_time(avg > 0 ? remaining / avg : 1e18).c_str(), fmt_targets().c_str());
            fflush(stdout);
        }
    }
    g_stop = true;
    for (auto &t : th) t.join();

    bool exhausted = safe_index() >= g_nchunks;
    if (exhausted) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::lock_guard<std::mutex> lp(g_print_mtx);
        printf("\r\033[K%s cles (%llu) | 100.0000%% | ecoule %s%s",
               fmt_keys((double)g_count.load()).c_str(), (unsigned long long)g_count.load(),
               fmt_time(el).c_str(), fmt_targets().c_str());
        fflush(stdout);
    }
    print_final_summary(exhausted, true);
    return 0;
#else
    u128 lo = g_lo;
    u128 span = g_hi - g_lo + (u128)1; // nombre réel de clés à couvrir (pas forcément une puissance de deux)
    {
        // Correctif (bug présent depuis la v23) : le nombre de chunks est stocké dans un uint64_t.
        // Au-delà de 2^62 chunks (span/2^cpu_chunk_bits), il débordait silencieusement (ex: puzzle 80 avec
        // cpu_chunk_bits=10 -> 2^69 chunks tronqué à 0, donc 1 seul chunk scanné puis « plage entièrement
        // scannée » à tort). On agrandit alors la taille de chunk pour rester sous 2^62 chunks.
        int span_bits = u128_bitlen(span - (u128)1);
        if (span_bits - cpu_chunk_bits > 62) {
            cpu_chunk_bits = span_bits - 62;
            std::cerr << "Note : intervalle très large, taille de chunk portée à 2^" << cpu_chunk_bits << ".\n";
        }
        u128 chunk_size = (u128)1 << cpu_chunk_bits;
        uint64_t nchunks_real = (uint64_t)((span + chunk_size - (u128)1) / chunk_size);
        g_nchunks = next_pow2_u64(nchunks_real == 0 ? 1 : nchunks_real);
    }

    std::random_device rd;
    uint64_t stride = seq ? 1 : ((((uint64_t)rd() << 32) | rd()) | 1);

    g_cur.assign(nt, IDLE);

    std::cout << "Plage : " << to_hex(lo) << " -> " << to_hex(g_hi)
              << " | cibles : " << g_ts.size()
              << " | threads : " << nt << " | mode : " << (seq ? "séquentiel" : "aléatoire sans doublon")
              << " | chunk : 2^" << cpu_chunk_bits << " | implémentation pure (sans bibliothèque)\n";

    g_count = 0; // l'auto-test ne doit rien laisser dans le compteur
    std::vector<std::thread> th;
    for (unsigned i = 0; i < nt; ++i) th.emplace_back(worker, i, lo, span, cpu_chunk_bits, stride);

    auto t0 = std::chrono::steady_clock::now(), tl = t0;
    uint64_t last = 0;
    const double total = (double)span;
    auto all_idle = [&] {
        std::lock_guard<std::mutex> l(g_mtx);
        if (g_next < g_nchunks) return false;
        for (auto c : g_cur) if (c != IDLE) return false;
        return true;
    };

    int sleep_ms = std::max(50, (int)(g_display_interval * 1000));
    // On attend par petits pas de 50 ms (et non d'un seul bloc de 'sleep_ms', qui peut valoir
    // jusqu'à 30 s ou plus avec -s) afin qu'un Ctrl-C soit pris en compte quasi immédiatement :
    // sans ça, g_stop passait bien à true tout de suite (les threads s'arrêtaient vite), mais le
    // thread principal restait endormi jusqu'à la fin de l'intervalle d'affichage avant de s'en
    // apercevoir, joindre les threads et rendre la main — ce qui donnait l'impression d'un
    // programme qui ne répond pas au Ctrl-C.
    const int poll_ms = 50;
    int since_print_ms = sleep_ms; // force un premier affichage dès le premier passage
    while (!g_stop && !all_idle()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
        since_print_ms += poll_ms;
        if (g_stop || all_idle()) break;
        if (since_print_ms < sleep_ms) continue;
        since_print_ms = 0;
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - t0).count();
        double dt = std::chrono::duration<double>(now - tl).count();
        uint64_t c = g_count.load();
        double inst = (c - last) / dt, avg = c / el;
        last = c; tl = now;
        double remaining = total - (double)c;
        {
            std::lock_guard<std::mutex> lp(g_print_mtx);
            printf("\r\033[K%s clés | %s clés/s | %.4f%% | écoulé %s | ETA max %s%s",
                   fmt_keys((double)c).c_str(), fmt_keys(inst).c_str(), 100.0 * (total - remaining) / total,
                   fmt_time(el).c_str(), fmt_time(avg > 0 ? remaining / avg : 1e18).c_str(),
                   fmt_targets().c_str());
            fflush(stdout);
        }
    }
    g_stop = true;
    for (auto &t : th) t.join();

    bool exhausted = safe_index() >= g_nchunks;
    if (exhausted) {
        // Dernière ligne de progression : sans elle, l'affichage resterait figé sur la dernière
        // mesure périodique (ex: 92 %) alors que la plage est terminée.
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::lock_guard<std::mutex> lp(g_print_mtx);
        printf("\r\033[K%s clés (%llu) | 100.0000%% | écoulé %s%s",
               fmt_keys((double)g_count.load()).c_str(), (unsigned long long)g_count.load(),
               fmt_time(el).c_str(), fmt_targets().c_str());
        fflush(stdout);
    }
    print_final_summary(exhausted, true);
#endif
}
