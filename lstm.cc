#include "/home/wentingj/tensorflow/tensorflow/core/framework/op.h"
#include "/home/wentingj/tensorflow/tensorflow/core/framework/shape_inference.h"

#include "/home/wentingj/tensorflow/tensorflow/core/framework/op_kernel.h"

#include "mkl.h"
#include "math.h"
using namespace std;
using namespace tensorflow;

REGISTER_OP("IntelLstm")
    .Input("x: float")
    .Input("w_x: float")
    .Input("w_h: float")
    .Input("b: float")
    .Input("h_0: float")
    .Input("c_0: float")
    //.Input("return_sequences: bool")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
     //c->set_output(0,c->input(0));
     return Status::OK();
    });


class IntelLstmOp : public OpKernel {
 public:
   explicit IntelLstmOp(OpKernelConstruction* context) : OpKernel(context) {}
     void Compute(OpKernelContext* context) override {
         int i,j,p;
         //x
         const Tensor* input_tensor = &context->input(0);
         auto input_flat = input_tensor->flat<float>();
         auto x = input_flat.data();
         printf( " run to %s on line %d\n " ,__FILE__, __LINE__);
         
         //w_x
         const Tensor* w_x_tensor = &context->input(1);
         auto w_x_flat = w_x_tensor->flat<float>();
         auto w_x = w_x_flat.data();

         //w_h
         const Tensor* w_h_tensor = &context->input(2);
         auto w_h_flat = w_h_tensor->flat<float>();
         auto w_h = w_h_flat.data();

         //b
         const Tensor* b_tensor = &context->input(3);
         auto b_flat = b_tensor->flat<float>();
         auto b = b_flat.data();

         //h_0
         const Tensor* h_0_tensor = &context->input(4);
         auto h_0_flat = h_0_tensor->flat<float>();
         auto h_0 = h_0_flat.data();
       
         //c_0
         const Tensor* c_0_tensor = &context->input(5);
         auto c_0_flat = c_0_tensor->flat<float>();
         auto c_0 = c_0_flat.data();

         ////return_sequences
         //const Tensor* return_sequences_tensor = &context->input(6);
         //auto return_sequences_flat = return_sequences_tensor->flat<bool>();
         //auto return_sequences = return_sequences_flat.data();

         printf( " run to %s on line %d\n " ,__FILE__, __LINE__);
         int max_len = 128;//max timestep
         int batch_size = input_tensor->shape().dim_size(2);
         printf("batch_size=%d\n", batch_size);
         int time_step = input_tensor->shape().dim_size(0);
         printf("time_step=%d\n", time_step);
         int input_dim = input_tensor->shape().dim_size(1);
         printf("input_dim=%d\n", input_dim);
         int hid = w_h_tensor->shape().dim_size(1);
         printf("hid=%d\n", hid);
    bool return_sequences = true;

         //output
         TensorShape shapeY = TensorShape({time_step,hid,batch_size});
         Tensor* output_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(0, shapeY, &output_tensor));
         auto output_flat = output_tensor->flat<float>();
         auto y = output_flat.data();

         printf( " run to %s on line %d\n " ,__FILE__, __LINE__);
         //temp buf
         const float** A = (const float**)mkl_malloc(4 * max_len * sizeof (float*), 64);
         const float** B = (const float**)mkl_malloc(4 * max_len * sizeof (float*), 64);
         float** C = (float**)mkl_malloc(4 * max_len * sizeof (float*), 64);
         float* x_temp = (float*)mkl_malloc(max_len * 4 * batch_size * hid * sizeof (float), 64);
         float* f_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);
         float* i_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);
         float* c_wave_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);
         float* o_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);
         float* c_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);

         // w_x * x
         MKL_INT m[1]; 
         MKL_INT n[1]; 
         MKL_INT k[1]; 
         
         MKL_INT lda[1]; 
         MKL_INT ldb[1]; 
         MKL_INT ldc[1]; 
         
         CBLAS_TRANSPOSE transA[1]; 
         CBLAS_TRANSPOSE transB[1]; 
         
         float alpha[1]; 
         float beta[1]; 
         MKL_INT size_per_grp[1]; 
     
         m[0] = hid;
         k[0] = input_dim;
         n[0] = batch_size;
         
         lda[0] = k[0]; 
         ldb[0] = n[0]; 
         ldc[0] = n[0]; 
         
         transB[0] = CblasNoTrans; 
         transA[0] = CblasNoTrans; 
         
         alpha[0] = 1.0; 
         if (b == NULL) {
             beta[0] = 0.0;
         }
         else {
             beta[0] = 1.0;
             #pragma omp parallel for 
             for (i = 0; i < time_step; i++) { 
                 for (j = 0; j < batch_size; j++) { 
                     for (p = 0; p < hid; p++) { 
                         size_t offset0 = i * batch_size * hid + j * hid + p; 
                         size_t offset1 = (i + time_step) * batch_size * hid + j * hid + p; 
                         size_t offset2 = (i + 2 * time_step) * batch_size * hid + j * hid + p; 
                         size_t offset3 = (i + 3 * time_step) * batch_size * hid + j * hid + p; 
             
                         x_temp[offset0] = b[p]; 
                         x_temp[offset1] = b[p + hid]; 
                         x_temp[offset2] = b[p + 2 * hid]; 
                         x_temp[offset2] = b[p + 3 * hid]; 
                     } 
                 } 
             } 
         }
         printf( " run to %s on line %d\n " ,__FILE__, __LINE__);
         size_per_grp[0] = 4 * time_step;
     
         if (NULL == A || NULL == B || NULL == C || NULL == x_temp || NULL == f_t || NULL == i_t || NULL == c_wave_t || NULL == o_t || NULL == c_t) {
             printf( "\n ERROR: malloc global buffers failed \n\n");
             return;
         }
         #pragma omp parallel for 
         for (i = 0; i < time_step; i++) { 
             A[i] = w_x;                                       // w_fx
             A[i + time_step] = w_x + input_dim * hid;         // w_ix
             A[i + 2 * time_step] = w_x + 2 * input_dim * hid; // w_cx 
             A[i + 3 * time_step] = w_x + 3 * input_dim * hid; // w_ox 
         
             B[i] = x + i * k[0] * n[0]; 
             B[i + time_step] = B[i]; 
             B[i + 2 * time_step] = B[i]; 
             B[i + 3 * time_step] = B[i]; 
         
             C[i] = x_temp + i * m[0] * n[0]; 
             C[i + time_step] = x_temp + (i + time_step) * m[0] * n[0]; 
             C[i + 2 * time_step] = x_temp + (i + 2 * time_step) * m[0] * n[0]; 
             C[i + 3 * time_step] = x_temp + (i + 3 * time_step) * m[0] * n[0]; 
         } 
         cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp); 
     
         // loop on step
         m[0] = hid;
         k[0] = hid;
         n[0] = batch_size;
         
         beta[0] = 1.0;
     
         lda[0] = k[0]; 
         ldb[0] = n[0]; 
         ldc[0] = n[0]; 
         size_per_grp[0] = 4;
         
         A[0] = w_h;                //w_fh
         A[1] = w_h + hid * hid;    //w_ih
         A[2] = w_h + 2 * hid * hid;//w_ch
         A[3] = w_h + 3 * hid * hid;//w_oh
         
         B[0] = h_0;
         B[1] = h_0;
         B[2] = h_0;
         B[3] = h_0;
     
         size_t mn = m[0] * n[0];
         #pragma omp parallel for
         for (j = 0; j < mn; j++) {
             c_t[j] = c_0[j];
         }
     
         for (i = 0; i < time_step; i++) {
             // f,i,c_wave,o
             C[0] = x_temp + i * m[0] * n[0];
             C[1] = x_temp + (i + time_step) * m[0] * n[0];
             C[2] = x_temp + (i + 2 * time_step) * m[0] * n[0];
             C[3] = x_temp + (i + 3 * time_step) * m[0] * n[0];
     
             cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
     
             // sigmoid for f,i,o, tanh for c_wave
             #pragma omp parallel for
             for (j = 0; j < mn; j++) {
                 float exp_f = exp((float)(C[0][j]));
                 float exp_i = exp((float)(C[1][j]));
                 c_wave_t[j] = tanh((float)(C[2][j]));
                 float exp_o = exp((float)(C[3][j]));
                 f_t[j] = exp_f / ((float)1.0 + exp_f);        
                 i_t[j] = exp_i / ((float)1.0 + exp_i);
                 o_t[j] = exp_o / ((float)1.0 + exp_o);
             }
             //c
             #pragma omp parallel for 
             for (j = 0; j < mn; j++) { 
                 c_t[j] = (float)((float)(f_t[j]) * (float)(c_t[j]) + (float)(i_t[j]) * (float)(c_wave_t[j])); 
             }
             //h
             float* y_ptr = NULL;
             if (return_sequences) {
                 y_ptr = y + i * batch_size * hid;
             } else {
                 y_ptr = y;
             }
             #pragma omp parallel for
             for (j = 0; j < mn; j++) {
                 y_ptr[j] = (float)(o_t[j]) * tanh((float)(c_t[j]));
             }
             // update
             B[0] = y_ptr;
             B[1] = B[0];
             B[2] = B[0];
             B[3] = B[0];
         }
         printf( " run to %s on line %d\n " ,__FILE__, __LINE__);
         mkl_free(A);
         mkl_free(B);
         mkl_free(C);
         mkl_free(x_temp);
         mkl_free(f_t);
         mkl_free(i_t);
         mkl_free(c_wave_t);
         mkl_free(o_t);
         mkl_free(c_t);
   }
};

REGISTER_KERNEL_BUILDER(Name("IntelLstm").Device(DEVICE_CPU), IntelLstmOp);
