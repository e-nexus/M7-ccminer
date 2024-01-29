/*
 * m7 algorithm
 *
 */


extern "C"
{
#include "sph/sph_sha2.h"
#include "sph/sph_keccak.h"
#include "sph/sph_ripemd.h"
#include "sph/sph_haval.h"
#include "sph/sph_tiger.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_blake.h"
}
#include "miner.h"
#include "cuda_helper.h"

extern int shortdevice_map[MAX_GPUS];

static uint64_t *d_hash[MAX_GPUS];
static uint64_t *KeccakH[MAX_GPUS];
static uint64_t *Sha512H[MAX_GPUS];
static uint64_t *d_prod0[MAX_GPUS];
static uint64_t *d_prod1[MAX_GPUS];

extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

extern uint32_t m7_sha256_cpu_hash_300(int thr_id, int threads, uint32_t startNounce, uint64_t *d_nonceVector, uint64_t *d_hash, int order);

extern void m7_sha256_setBlock_120(void *data,const void *ptarget);
extern void m7_sha256_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order);
extern void m7_sha256_cpu_init(int thr_id, int threads);

extern void m7_sha512_cpu_init(int thr_id, int threads);
extern void m7_sha512_setBlock_120(void *pdata);
extern void m7_sha512_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_hash, int order);

extern void m7_ripemd160_cpu_init(int thr_id, int threads);
extern void m7_ripemd160_setBlock_120(void *pdata);
extern void m7_ripemd160_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_hash, int order);

extern void tiger192_cpu_init(int thr_id, int threads);
extern void tiger192_setBlock_120(void *pdata);
extern void m7_tiger192_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_hash, int order);

extern void m7_bigmul_init(int thr_id, int threads);
extern void m7_bigmul_unroll1_cpu(int thr_id, int threads,uint64_t* Hash1, uint64_t* Hash2,uint64_t *finalHash,int order);
extern void m7_bigmul_unroll2_cpu(int thr_id, int threads,uint64_t* Hash1, uint64_t* Hash2,uint64_t *finalHash,int order);

extern void cpu_mul(int thr_id, int threads, uint32_t alegs, uint32_t blegs, uint64_t *g_a, uint64_t *g_b, uint64_t *g_p, int order);
extern void cpu_mulT4(int thr_id, int threads, uint32_t alegs, uint32_t blegs, uint64_t *g_a, uint64_t *g_b, uint64_t *g_p, int order);
extern void mul_init();

extern void m7_keccak512_setBlock_120(void *pdata);
extern void m7_keccak512_cpu_hash(int thr_id, int threads, uint32_t startNounce, uint64_t *d_hash, int order);
extern void m7_keccak512_cpu_init(int thr_id, int threads);

extern void m7_whirlpool512_cpu_init(int thr_id, int threads, int flag);
extern void m7_whirlpool512_setBlock_120(void *pdata);
extern void m7_whirlpool512_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order);

extern void haval256_setBlock_120(void *data);
extern void m7_haval256_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order);

float tp_coef_f[8] = {1, 1, 1, 1, 1, 1, 1, 1};
extern bool opt_benchmark;

extern "C" int scanhash_m7(int thr_id, uint32_t *pdata, const uint32_t *ptarget, uint32_t max_nonce, unsigned long  *hashes_done) {
	if (opt_benchmark) ((uint32_t*)ptarget)[7] = 0x0000ff;
	int throughput = 256 * 256 * 100;
	const uint32_t FirstNonce = pdata[29];
	static bool init[8] = {0,0,0,0,0,0,0,0};

	if (!init[thr_id]) {
		cudaSetDevice(device_map[thr_id]);
		cudaDeviceReset();
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

		cudaMalloc(&d_prod0[thr_id],      35 *sizeof(uint64_t) * throughput*tp_coef_f[thr_id]);
		cudaMalloc(&d_prod1[thr_id],      38 *sizeof(uint64_t) * throughput*tp_coef_f[thr_id]);
		cudaMalloc(&KeccakH[thr_id],     8 *sizeof(uint64_t) * throughput*tp_coef_f[thr_id]);

		m7_sha256_cpu_init(thr_id, throughput*tp_coef_f[thr_id]);
		m7_sha512_cpu_init(thr_id, throughput*tp_coef_f[thr_id]);
		m7_keccak512_cpu_init(thr_id, throughput*tp_coef_f[thr_id]);
        tiger192_cpu_init(thr_id, throughput*tp_coef_f[thr_id]);
		m7_whirlpool512_cpu_init(thr_id, throughput*tp_coef_f[thr_id],0);
		m7_ripemd160_cpu_init(thr_id, throughput*tp_coef_f[thr_id]);
		m7_bigmul_init(thr_id, throughput*tp_coef_f[thr_id]);
		mul_init();
		init[thr_id] = true;
	}

	if (pdata[28] == 0) pdata[28] = rand();	// multi-rig solo workaround

	m7_whirlpool512_setBlock_120((void*)pdata);
    m7_sha256_setBlock_120((void*)pdata,ptarget);
	m7_sha512_setBlock_120((void*)pdata);
	haval256_setBlock_120((void*)pdata);
	m7_keccak512_setBlock_120((void*)pdata);
	m7_ripemd160_setBlock_120((void*)pdata);
	tiger192_setBlock_120((void*)pdata);

	do {
		int order = 0;

		m7_keccak512_cpu_hash(thr_id, throughput*tp_coef_f[thr_id], pdata[29], KeccakH[thr_id], order++);
		m7_sha512_cpu_hash_120(thr_id, throughput*tp_coef_f[thr_id], pdata[29], d_prod1[thr_id], order++);
        cpu_mulT4(0, throughput*tp_coef_f[thr_id], 8, 8, d_prod1[thr_id], KeccakH[thr_id], d_prod0[thr_id],order); //64

        m7_whirlpool512_cpu_hash_120(thr_id, throughput*tp_coef_f[thr_id], pdata[29], KeccakH[thr_id], order++);
		cpu_mulT4(0, throughput*tp_coef_f[thr_id],8, 16, KeccakH[thr_id], d_prod0[thr_id], d_prod1[thr_id],order); //128

		m7_sha256_cpu_hash_120(thr_id, throughput*tp_coef_f[thr_id], pdata[29], KeccakH[thr_id], order++);
		cpu_mulT4(0, throughput*tp_coef_f[thr_id], 4, 24, KeccakH[thr_id], d_prod1[thr_id], d_prod0[thr_id],order); //96

		m7_haval256_cpu_hash_120(thr_id, throughput*tp_coef_f[thr_id], pdata[29], KeccakH[thr_id], order++);
		cpu_mulT4(0, throughput*tp_coef_f[thr_id], 4, 28, KeccakH[thr_id], d_prod0[thr_id], d_prod1[thr_id],order);  //112

		m7_tiger192_cpu_hash_120(thr_id, throughput*tp_coef_f[thr_id], pdata[29], KeccakH[thr_id], order++);
		m7_bigmul_unroll1_cpu(thr_id, throughput*tp_coef_f[thr_id], KeccakH[thr_id], d_prod1[thr_id], d_prod0[thr_id],order);

		m7_ripemd160_cpu_hash_120(thr_id, throughput*tp_coef_f[thr_id], pdata[29], KeccakH[thr_id], order++);
		m7_bigmul_unroll2_cpu(thr_id, throughput*tp_coef_f[thr_id], KeccakH[thr_id], d_prod0[thr_id], d_prod1[thr_id],order);

		uint32_t foundNonce = m7_sha256_cpu_hash_300(thr_id, throughput*tp_coef_f[thr_id], pdata[29], NULL, d_prod1[thr_id], order);
		if  (foundNonce != 0xffffffff) {
			*hashes_done = pdata[29] - FirstNonce + throughput*tp_coef_f[thr_id];
            pdata[29] = foundNonce;
			return 1;
        }

		pdata[29] += throughput*tp_coef_f[thr_id];

	} while (((uint64_t)max_nonce > ((uint64_t)(pdata[29]) + (uint64_t)throughput*tp_coef_f[thr_id])) && !work_restart[thr_id].restart);

	*hashes_done = pdata[29] - FirstNonce;
	return 0;
}
