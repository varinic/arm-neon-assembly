#include <stdlib.h>
#include <stdio.h>
#include <arm_neon.h>


void fill_num_inc(int8_t * a, const size_t m, int8_t val){
  for(size_t i =0;i<m;i++){
    *(a+i)=val+i;
  }
}

void fill_num_int(int * a, const size_t m, int val){
  for(size_t i =0;i<m;i++){
    *(a+i)=val;
  }
}
void fill_num(int8_t * a, const size_t m, int8_t val){
  for(size_t i =0;i<m;i++){
    *(a+i)=val;
  }
}

void print(int8_t * a,size_t m){
  for(size_t i=0;i<m;i++){
    printf("%d ",(int)(*(a+i)));
  }
  printf("\n");
}

void print_int(int * a,size_t m){
  for(size_t i=0;i<m;i++){
    printf("%d ",(int)(*(a+i)));
  }
  printf("\n");
}

void nor_mul_test(const size_t m, int8_t * a, int8_t * b, int * c){
  for(size_t i=0;i<m;i++){
    c[i]=a[i]*b[i];
  }
}

void neon_mul_test(const size_t m, int8_t * a, int8_t * b, int * c){
  int8x16_t va,vb;
  int16x8_t vt[2];
  int32x4_t vc[4]={0,0,0,0};
  va=vld1q_s8( a );
  vb=vld1q_s8( b );
  vt[0]=vmull_s8(vget_low_s8(va),vget_low_s8(vb));
  vt[1]=vmull_s8(vget_high_s8(va),vget_high_s8(vb));
  vc[0]=vaddw_s16(vc[0],vget_low_s16(vt[0]));
  vc[1]=vaddw_s16(vc[1],vget_high_s16(vt[0]));
  vc[2]=vaddw_s16(vc[2],vget_low_s16(vt[1]));
  vc[3]=vaddw_s16(vc[3],vget_high_s16(vt[1]));
  for(size_t i=0;i<m/4;i++){
    vst1q_s32(c+i*4,vc[i]);
  }
}

void asm_mul_test(const size_t m, int8_t * a, int8_t * b, int * c){
#ifdef __aarch64__
  __asm__ __volatile__(
        
        "mov x4, %[dim_m]\n\t"
        "mov x5, %[addr_a]\n\t"
        "mov x6, %[addr_b]\n\t"
        "mov x7, %[addr_c]\n\t"

        "movi v5.2d, #0\n\t"
        "movi v6.2d, #0\n\t"
        "movi v7.2d, #0\n\t"
        "movi v8.2d, #0\n\t"

        "ld1 {v1.2d}, [x5], #16\n\t"
        "ld1 {v2.2d}, [x6], #16\n\t"
        "smull  v3.8h, v1.8b,  v2.8b\n\t"
        "smull2 v4.8h, v1.16b, v2.16b\n\t"
        "saddw  v5.4s, v5.4s, v3.4h\n\t"
        "saddw2 v6.4s, v6.4s, v3.8h\n\t"
        "saddw  v7.4s, v7.4s, v4.4h\n\t"
        "saddw2 v8.4s, v8.4s, v4.8h\n\t"
        "st1 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[addr_c]]\n\t"
        : // output
        : // input
        [dim_m] "r" (m/16*16),
        [addr_a] "r" (a), [addr_b] "r" (b), [addr_c] "r" (c)
        : //clobber
        "cc", "memory" , "x4", "x5", "x6", "x7",
        "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"

  );

#else
  printf("not aarch64");
#endif

}

int main(){
  const size_t M = 16;
  int8_t A[M],B[M];
  int C[M];
  fill_num_inc(A, M, 1);
  fill_num_inc(B, M, 2);
  printf(" Array A : ");
  print(A, M);
  printf(" Array B : ");
  print(B, M);
  //
  nor_mul_test(M,A,B,C);
  printf(" After nor_mul_test Array C : ");
  print_int(C, M);
  //
  fill_num_int(C, M, -1);
  neon_mul_test(M,A,B,C);
  printf(" After neon_mul_test Array C : ");
  print_int(C, M);
  //
  fill_num_int(C, M, -1);
  asm_mul_test(M,A,B,C);
  printf(" After asm_mul_test Array C : ");
  print_int(C, M);

  return 0;
}
