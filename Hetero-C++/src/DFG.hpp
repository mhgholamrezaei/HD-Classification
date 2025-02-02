#pragma once

#include <hpvm_hdc.h>
#include <heterocc.h>
#include <iostream>

//#define HAMMING_DIST

#undef D
#undef N_FEATURES
#undef K

typedef int binary;
#ifndef ACCEL 
typedef float hvtype;
#else
typedef int16_t hvtype;
#endif



template <int N, typename elemTy>
void print_hv_alt(__hypervector__<N, elemTy> hv) {
    int num_neg_one = 0;
    int num_one = 0;
    std::cout << "[";
    for (int i = 0; i < N-1; i++) {
        std::cout << hv[0][i] << ", ";
        if(hv[0][i] == 1){
            num_one ++;
        } else if(hv[0][i] == -1){
            num_neg_one ++;
        }
    }
    std::cout << hv[0][N-1] << "]\n";

    std::cout <<"# Negative 1: "<< num_neg_one <<", # Positive 1: "<< num_one << "\n";
    return;
}


void ptr_print_hv_alt(hvtype* ptr, int num_elems){
    std::cout << "[ ";
    for(int i = 0; i < num_elems; i++){
        std::cout <<ptr[i] <<" ";
    }
    std::cout<<"\n";
}

// RANDOM PROJECTION ENCODING!!
// Matrix-vector mul
// Encodes a single vector using a random projection matrix
//
// RP encoding reduces N_features -> D 

template <typename T>
T zero_hv(size_t loop_index_var) {
	return 0;
}

template<int D, int N_FEATURES>
void rp_encoding_node(/* Input Buffers: 2*/
        __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
        __hypervector__<N_FEATURES, hvtype>* input_datapoint_ptr, size_t input_datapoint_size, // __hypervector__<N_FEATURES, int> 
        /* Output Buffers: 1*/
        __hypervector__<D, hvtype>* output_hv_ptr, size_t output_hv_size) { // __hypervector__<D, binary>
    
    void* section = __hetero_section_begin();


    void* task = __hetero_task_begin(
        /* Input Buffers: 2*/ 3, rp_matrix_ptr, rp_matrix_size, input_datapoint_ptr, input_datapoint_size, output_hv_ptr, output_hv_size,
        /* Parameters: 0*/
        /* Output Buffers: 1*/ 1, output_hv_ptr, output_hv_size,
        "inner_rp_encoding_task"
    );

    __hetero_hint(DEVICE);

    __hypervector__<D, hvtype> encoded_hv = __hetero_hdc_create_hypervector<D, hvtype>(0, (void*) zero_hv<hvtype>);
    *output_hv_ptr = encoded_hv;
    encoded_hv = __hetero_hdc_matmul<D, N_FEATURES, hvtype>(*input_datapoint_ptr, *rp_matrix_ptr); 
    *output_hv_ptr = encoded_hv;

    __hetero_task_end(task); 

    __hetero_section_end(section);
    return;
}

template<int D, int N_FEATURES>
void rp_encoding_node_copy(/* Input Buffers: 2*/
        __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
        __hypervector__<N_FEATURES, hvtype>* input_datapoint_ptr, size_t input_datapoint_size, // __hypervector__<N_FEATURES, int> 
        /* Output Buffers: 1*/
        __hypervector__<D, hvtype>* output_hv_ptr, size_t output_hv_size) { // __hypervector__<D, binary>
    
    void* section = __hetero_section_begin();
    void* task = __hetero_task_begin(
        /* Input Buffers: 2*/ 3, rp_matrix_ptr, rp_matrix_size, input_datapoint_ptr, input_datapoint_size, output_hv_ptr, output_hv_size,
        /* Parameters: 0*/
        /* Output Buffers: 1*/ 1, output_hv_ptr, output_hv_size,
        "inner_rp_encoding_task"
    );

    __hetero_hint(DEVICE);
    
    __hypervector__<D, hvtype> encoded_hv = __hetero_hdc_create_hypervector<D, hvtype>(0, (void*) zero_hv<hvtype>);
    *output_hv_ptr = encoded_hv;

    encoded_hv = __hetero_hdc_matmul<D, N_FEATURES, hvtype>(*input_datapoint_ptr, *rp_matrix_ptr); 
    *output_hv_ptr = encoded_hv;
    __hetero_task_end(task); 

    __hetero_section_end(section);
    return;
}

template<int D, int N_FEATURES>
void InitialEncodingDFG(
        /* Input Buffers: 2*/
        __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
        __hypervector__<N_FEATURES, hvtype>* input_datapoint_ptr, size_t input_datapoint_size, // __hypervector__<N_FEATURES, int> 
        /* Output Buffers: 1*/
        __hypervector__<D, hvtype>* output_hv_ptr, size_t output_hv_size) { // __hypervector__<D, binary>
    
    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(
        /* Input Buffers: 2*/ 3, rp_matrix_ptr, rp_matrix_size, input_datapoint_ptr, input_datapoint_size, output_hv_ptr, output_hv_size,
        /* Parameters: 0*/
        /* Output Buffers: 1*/ 1, output_hv_ptr, output_hv_size,
        "initial_encoding_wrapper"
    );

    // Specifies that the following node is performing an HDC Encoding step
    __hetero_hdc_encoding(6, (void*) rp_encoding_node_copy<D, N_FEATURES>, rp_matrix_ptr, rp_matrix_size, input_datapoint_ptr, input_datapoint_size, output_hv_ptr, output_hv_size);

    __hetero_task_end(task); 

    __hetero_section_end(section);
    return;
}

// RP Matrix Node Generation. Performs element wise rotation on ID matrix rows and stores into destination buffer.
template <int D,  int N_FEATURES>
void gen_rp_matrix(/* Input Buffers*/
        __hypervector__<D, hvtype>* rp_seed_vector, size_t  rp_seed_vector_size,
        __hypervector__<D, hvtype>* row_buffer, size_t  row_buffer_size,
        __hypermatrix__<N_FEATURES, D, hvtype>* shifted_matrix, size_t  shifted_matrix_size,
        __hypermatrix__<D, N_FEATURES, hvtype>* transposed_matrix, size_t  transposed_matrix_size
        ){


    void* root_section = __hetero_section_begin();


    void* root_task = __hetero_task_begin(
            /* Num Inputs */ 4,
         rp_seed_vector,   rp_seed_vector_size,
         shifted_matrix,   shifted_matrix_size,
         row_buffer, row_buffer_size,
         transposed_matrix,   transposed_matrix_size,
         /* Num Outputs*/ 1,
         transposed_matrix,   transposed_matrix_size,
        "gen_root_task"
    );

    void* wrapper_section = __hetero_section_begin();


    {
    void* gen_shifted_task = __hetero_task_begin(
         /* Num Inputs */ 3,
         rp_seed_vector,   rp_seed_vector_size,
         shifted_matrix,   shifted_matrix_size,
         row_buffer, row_buffer_size,
         /* Num Outputs */ 1,
         shifted_matrix,   shifted_matrix_size,
         "gen_shifted_matrix_task");

    __hetero_hint(DEVICE);

    for (int i = 0; i < N_FEATURES; i++) {
    	__hypervector__<D, hvtype>  row = __hetero_hdc_wrap_shift<D, hvtype>(*rp_seed_vector, i);
        *row_buffer = row;
        __hetero_hdc_set_matrix_row<N_FEATURES, D, hvtype>(*shifted_matrix, row, i);
    } 
   __hetero_task_end(gen_shifted_task); 
   }

   {
    void* transpose_task = __hetero_task_begin(
         /* Num Inputs */ 2,
         shifted_matrix,   shifted_matrix_size,
         transposed_matrix,   transposed_matrix_size,
         /* Num Outputs */ 1,
         transposed_matrix,   transposed_matrix_size,
         "gen_tranpose_task");
    __hetero_hint(DEVICE);
    *transposed_matrix = __hetero_hdc_matrix_transpose<N_FEATURES, D, hvtype>(*shifted_matrix, N_FEATURES, D);
    __hetero_task_end(transpose_task); 
   }


   __hetero_section_end(wrapper_section);

   __hetero_task_end(root_task); 

   __hetero_section_end(root_section);
}




template<int D, int N_FEATURES>
void rp_encoding_node_copy_copy(/* Input Buffers: 2*/
        __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
        __hypervector__<N_FEATURES, hvtype>* input_datapoint_ptr, size_t input_datapoint_size, // __hypervector__<N_FEATURES, int> 
        /* Output Buffers: 1*/
        __hypervector__<D, hvtype>* output_hv_ptr, size_t output_hv_size) { // __hypervector__<D, binary>
    
    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(
        /* Input Buffers: 2*/ 3, rp_matrix_ptr, rp_matrix_size, input_datapoint_ptr, input_datapoint_size, output_hv_ptr, output_hv_size,
        /* Parameters: 0*/
        /* Output Buffers: 1*/ 1, output_hv_ptr, output_hv_size,
        "inner_rp_encoding_task"
    );

    __hetero_hint(DEVICE);
    
    __hypervector__<D, hvtype> encoded_hv = __hetero_hdc_create_hypervector<D, hvtype>(0, (void*) zero_hv<hvtype>);
    *output_hv_ptr = encoded_hv;

    encoded_hv = __hetero_hdc_matmul<D, N_FEATURES, hvtype>(*input_datapoint_ptr, *rp_matrix_ptr); 
    *output_hv_ptr = encoded_hv;

    __hetero_task_end(task); 

    __hetero_section_end(section);
    return;
}


// In the streaming implementation, this runs for each encoded HV, so N_VEC * EPOCHs times.

/* Just make guesses based on cossim  */
template<int D, int K, int N_VEC> // ONLY RUNS ONCE
void __attribute__ ((noinline)) classification_node_inference(
    __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, // __hypervector__<D, binary>
    __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, // __hypermatrix__<K, D, binary>
    __hypervector__<K, hvtype>* scores_ptr, size_t scores_size, // Used as Local var.
    __hypervector__<K, hvtype>* norms_ptr, size_t norms_size, // Used as Local var.
    int encoded_hv_idx,
    int* label_ptr, size_t label_size ) {   
    // Read classes hvs from host.

     void* section = __hetero_section_begin();

    void* task1 = __hetero_task_begin(
        /* Input Buffers: */ 4, encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, scores_ptr, scores_size, norms_ptr, norms_size, 
        /* Output Buffers: */ 1, scores_ptr, scores_size, "inference_calculate_score_task"
    );

    __hetero_hint(DEVICE);
    
    // Class HVs are created via 'clustering' on +1, -1 encoded hypervectors. (loop 269).
    __hypervector__<D, hvtype> encoded_hv = *encoded_hv_ptr;
    __hypermatrix__<K, D, hvtype> classes = *classes_ptr;

    __hypervector__<K, hvtype> scores = *scores_ptr; // Precision of these scores might need to be increased.

    #ifdef HAMMING_DIST
    *scores_ptr =  __hetero_hdc_hamming_distance<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr);
    #else
#ifndef ACCEL
    *norms_ptr = __hetero_hdc_l2norm<K, D, hvtype>(*classes_ptr);
    *scores_ptr = __hetero_hdc_matmul<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr); 
    *scores_ptr = __hetero_hdc_div<K, hvtype>(*scores_ptr, *norms_ptr);
    *scores_ptr = __hetero_hdc_absolute_value<K, hvtype>(*scores_ptr);
#endif
    #endif



    __hetero_task_end(task1);

    void* task2 = __hetero_task_begin(
        /* Input Buffers: 1*/ 4, scores_ptr, scores_size, label_ptr, label_size, encoded_hv_ptr, encoded_hv_size,
        /* paramters: 1*/       encoded_hv_idx,
        /* Output Buffers: 1*/ 1,  label_ptr, label_size, "inference_find_max_task"
    );  

    __hetero_hint(DEVICE);

    {
    __hypervector__<K, hvtype> scores = *scores_ptr;
    int max_idx = 0;
    
    hvtype* scores_elem_ptr = (hvtype*) scores_ptr;
    #ifdef HAMMING_DIST
    hvtype max_score = (hvtype) D - scores_elem_ptr[0]; // I think this is probably causing issues.
    #else
    hvtype max_score = (hvtype) scores_elem_ptr[0];
    #endif

    
    for (int k = 0; k < K; k++) {
        #ifdef HAMMING_DIST
        hvtype score = (hvtype) D - scores_elem_ptr[k];
        #else
        hvtype score = (hvtype) scores_elem_ptr[k];
        #endif
        if (score > max_score) {
            max_score = score;
            max_idx = k;
        }
    } 
    
    // Set the label to our guess 
    *label_ptr = max_idx; 
    }
    __hetero_task_end(task2);

    __hetero_section_end(section);
    return;
}



// Retraining epochs
// In the streaming implementation, this runs for each encoded HV, so N_VEC * EPOCHs times.
// classification_node is the hetero-c++ version of searchUnit from the original FPGA code.
template<int D, int K, int N_VEC>
void classification_node_training_rest(/* Input Buffers: 2 */
    __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size,
    __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, // Do we need a separate buffer or can we edit in place. We can edit in place. 
    __hypervector__<K, hvtype>* scores_ptr, size_t scores_size, // Used as Local var.
    __hypervector__<K, hvtype>* norms_ptr, size_t norms_size, // Used as Local var.
    __hypervector__<D, hvtype>* update_hv_ptr, size_t update_hv_size,  // Used in second stage of clustering node for extracting and accumulating
    int* argmax, size_t argmax_size,
    int label ) {
    


    void* section = __hetero_section_begin();

    void* task1 = __hetero_task_begin(
        /* Input Buffers: */ 4, encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, scores_ptr, scores_size, norms_ptr, norms_size, 
        /* Output Buffers: */ 1, scores_ptr, scores_size, "training_rest_scoring_task"
    );

    __hetero_hint(DEVICE);

    __hypervector__<D, hvtype> encoded_hv = *encoded_hv_ptr;
    __hypermatrix__<K, D, hvtype> classes = *classes_ptr;

    __hypervector__<K, hvtype> scores = *scores_ptr; // Precision of these scores might need to be increased.

    #ifdef HAMMING_DIST
    *scores_ptr =  __hetero_hdc_hamming_distance<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr);
    #else

#ifndef ACCEL
    *norms_ptr = __hetero_hdc_l2norm<K, D, hvtype>(*classes_ptr);
    *scores_ptr = __hetero_hdc_matmul<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr); 
    *scores_ptr = __hetero_hdc_div<K, hvtype>(*scores_ptr, *norms_ptr);
    *scores_ptr = __hetero_hdc_absolute_value<K, hvtype>(*scores_ptr);
#endif
    #endif



    __hetero_task_end(task1);

    void* task2 = __hetero_task_begin(
        /* Input Buffers: 1*/ 3, scores_ptr, scores_size,  classes_ptr, classes_size, argmax, argmax_size,
        /* paramters: 1*/      
        /* Output Buffers: 1*/ 2,  classes_ptr, classes_size, argmax, argmax_size, "training_rest_find_score_task"
    );  

    __hetero_hint(DEVICE);

    {
        __hypervector__<K, hvtype> scores = *scores_ptr;

        *argmax = 0;
        
        hvtype* scores_elem_ptr = (hvtype*) scores_ptr;
        #ifdef HAMMING_DIST
        hvtype max_score = (hvtype) D - scores_elem_ptr[0]; // I think this is probably causing issues.
        #else
        hvtype max_score = (hvtype) scores_elem_ptr[0];
        #endif
        
        for (int k = 0; k < K; k++) {
            #ifdef HAMMING_DIST
            hvtype score = (hvtype) D - scores_elem_ptr[k];
            #else
            hvtype score = (hvtype) scores_elem_ptr[k];
            #endif
            if (score > max_score) {
                max_score = score;
                *argmax = k;
            }
        } 
    }
    __hetero_task_end(task2);

    

    void* task3 = __hetero_task_begin(
        /* Input Buffers: 1*/ 5, encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, update_hv_ptr, update_hv_size, argmax, argmax_size,
        /* paramters: 1*/       label,
        /* Output Buffers: 1*/ 1,  classes_ptr, classes_size, "update_classes_task"
    );  

    __hetero_hint(DEVICE);

    int max_idx = *argmax;
    // Update the correct and mispredicted class
    if (label != max_idx) { // Incorrect prediction
        *update_hv_ptr =  __hetero_hdc_get_matrix_row<K, D, hvtype>(*classes_ptr, K, D, label);
        *update_hv_ptr = __hetero_hdc_sum<D, hvtype>(*update_hv_ptr, *encoded_hv_ptr); // May need an instrinsic for this.
        __hetero_hdc_set_matrix_row<K, D, hvtype>(*classes_ptr, *update_hv_ptr, label); // How do we normalize?

        *update_hv_ptr =  __hetero_hdc_get_matrix_row<K, D, hvtype>(*classes_ptr, K, D, max_idx);
        *update_hv_ptr = __hetero_hdc_sub<D, hvtype>(*update_hv_ptr, *encoded_hv_ptr); // May need an instrinsic for this.
        __hetero_hdc_set_matrix_row<K, D, hvtype>(*classes_ptr, *update_hv_ptr, max_idx); // How do we normalize?
    }
    __hetero_task_end(task3);

    __hetero_section_end(section);


    return;
}

// Dimensionality, Clusters, data point vectors, features per.
template <int D, int K, int N_VEC, int N_FEATURES>
void encoding_and_training_node( /* Input buffers: 3*/ 
                __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
                __hypervector__<N_FEATURES, hvtype>* datapoint_vec_ptr, size_t datapoint_vec_size, // Features
                 __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, // __hypermatrix__<K, D, binary> // Also an output.

                int label,
                /* Local Vars: 3 */
                __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, // // __hypervector__<D, binary>
                __hypervector__<K, hvtype>* scores_ptr, size_t scores_size,
                __hypervector__<K, hvtype>* norms_ptr, size_t norms_size,
                __hypervector__<D, hvtype>* update_hv_ptr, size_t update_hv_size,  // Used in second stage of clustering node for extracting and accumulating
                int* argmax_ptr, size_t argmax_size
                /* Parameters: 1*/
                /* Output Buffers: 1 (Classes)*/ 
                ){

    void* root_section = __hetero_section_begin();

    
    // Re-encode each iteration.
    void* encoding_task = __hetero_task_begin(
        /* Input Buffers: 3 */ 3, rp_matrix_ptr, rp_matrix_size, datapoint_vec_ptr, datapoint_vec_size, 
                                 encoded_hv_ptr, encoded_hv_size,
        /* Output Buffers: 1 */ 1, encoded_hv_ptr, encoded_hv_size,
        "training_encoding_task"  
    );

    rp_encoding_node<D, N_FEATURES>(rp_matrix_ptr, rp_matrix_size, datapoint_vec_ptr, datapoint_vec_size, encoded_hv_ptr, encoded_hv_size);

    __hetero_task_end(encoding_task);

    void* training_task = __hetero_task_begin(
        /* Input Buffers: 6 */  6 + 1, 
                                encoded_hv_ptr, encoded_hv_size, 
                                classes_ptr, classes_size, 
                                scores_ptr, scores_size,
                                norms_ptr, norms_size,
                                update_hv_ptr, update_hv_size,
                                argmax_ptr, argmax_size,
                                label,
        /* Output Buffers: 2 */ 1, classes_ptr, classes_size, 
        "training_task"  
    );

    classification_node_training_rest<D, K, N_VEC>(encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, scores_ptr, scores_size, norms_ptr, norms_size,  update_hv_ptr, update_hv_size, argmax_ptr, argmax_size, label); 

    __hetero_task_end(training_task);

    __hetero_section_end(root_section);
    return;
}

// Dimensionality, Clusters, data point vectors, features per.
template <int D, int K, int N_VEC, int N_FEATURES>
void training_root_node( /* Input buffers: 3*/ 
                __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
                __hypervector__<N_FEATURES, hvtype>* datapoint_vec_ptr, size_t datapoint_vec_size, // Features
                 __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, // __hypermatrix__<K, D, binary> // Also an output.
                /* Local Vars: 3 */
                __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, // // __hypervector__<D, binary>
                __hypervector__<K, hvtype>* scores_ptr, size_t scores_size,
                __hypervector__<K, hvtype>* norms_ptr, size_t norms_size,
                __hypervector__<D, hvtype>* update_hv_ptr, size_t update_hv_size,  // Used in second stage of clustering node for extracting and accumulating
                int* argmax_ptr, size_t argmax_size,
                /* Parameters: 1*/
                int label
                /* Output Buffers: 1 (Classes)*/ 
                ){
    void* root_section = __hetero_section_begin();

    // Re-encode each iteration.
    void* training_encoding_task = __hetero_task_begin(
        /* Input Buffers: 3 */ 9, 
            rp_matrix_ptr, rp_matrix_size, 
            datapoint_vec_ptr, datapoint_vec_size, 
            encoded_hv_ptr, encoded_hv_size,  
            classes_ptr, classes_size, 
            scores_ptr, scores_size,
            norms_ptr, norms_size,
            update_hv_ptr, update_hv_size,
            argmax_ptr, argmax_size,
            label,
        /* Output Buffers: 1 */ 1, 
        encoded_hv_ptr, encoded_hv_size,
        "training_encoding_task_wrapper"  
    );

    __hetero_hdc_training(
        18,
        (void*) encoding_and_training_node<D, K, N_VEC, N_FEATURES>,
        rp_matrix_ptr, rp_matrix_size, 
        datapoint_vec_ptr, datapoint_vec_size, 
        classes_ptr, classes_size, 
        label,
        encoded_hv_ptr, encoded_hv_size,  
        scores_ptr, scores_size,
        norms_ptr, norms_size,
        update_hv_ptr, update_hv_size,
        argmax_ptr, argmax_size
    );
    
    __hetero_task_end(training_encoding_task);

    __hetero_section_end(root_section);
    return;
}

// Dimensionality, Clusters, data point vectors, features per.
template <int D, int K, int N_VEC, int N_FEATURES>
void encoding_and_inference_node( /* Input buffers: 3*/ 
                __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
                __hypervector__<N_FEATURES, hvtype>* datapoint_vec_ptr, size_t datapoint_vec_size, // Features
                __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, // __hypermatrix__<K, D, binary>
                int* label_ptr, size_t label_size,
                /* Local Vars: 2*/
                __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, // // __hypervector__<D, binary>
                __hypervector__<K, hvtype>* scores_ptr, size_t scores_size,
                __hypervector__<K, hvtype>* norms_ptr, size_t norms_size,
                // FIXME, give scores its own type.
                /* Parameters: 1*/
                int encoded_hv_idx
                /* Output Buffers: 1*/
                ){

    void* root_section = __hetero_section_begin();

    // Re-encode each iteration.
    void* encoding_task = __hetero_task_begin(
        /* Input Buffers: 3 */ 3, rp_matrix_ptr, rp_matrix_size, datapoint_vec_ptr, datapoint_vec_size, 
                                encoded_hv_ptr, encoded_hv_size,
        /* Output Buffers: 1 */ 1, encoded_hv_ptr, encoded_hv_size,
        "inference_encoding_task"  
    );
    rp_encoding_node_copy_copy<D, N_FEATURES>(rp_matrix_ptr, rp_matrix_size, datapoint_vec_ptr, datapoint_vec_size, encoded_hv_ptr, encoded_hv_size);

    __hetero_task_end(encoding_task);

    void* inference_task = __hetero_task_begin(
        /* Input Buffers: 5 */  5 + 1, 
                                encoded_hv_ptr, encoded_hv_size, 
                                classes_ptr, classes_size, 
                                label_ptr, label_size,
                                scores_ptr, scores_size,
                                norms_ptr, norms_size,
        /* Parameters: 1 */     encoded_hv_idx,
        /* Output Buffers: 1 */ 1, label_ptr, label_size,
        "inference_task"  
    );

    classification_node_inference<D, K, N_VEC>(encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, scores_ptr, scores_size,norms_ptr, norms_size , encoded_hv_idx, label_ptr, label_size); 

    __hetero_task_end(inference_task);

    __hetero_section_end(root_section);
    return;
}

// Dimensionality, Clusters, data point vectors, features per.
template <int D, int K, int N_VEC, int N_FEATURES>
void inference_root_node( /* Input buffers: 3*/ 
                __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, // __hypermatrix__<N_FEATURES, D, binary>
                __hypervector__<N_FEATURES, hvtype>* datapoint_vec_ptr, size_t datapoint_vec_size, // Features
                __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, // __hypermatrix__<K, D, binary>
                /* Local Vars: 2*/
                __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, // // __hypervector__<D, binary>
                __hypervector__<K, hvtype>* scores_ptr, size_t scores_size,
                __hypervector__<K, hvtype>* norms_ptr, size_t norms_size,
                /* Parameters: 1*/
                int encoded_hv_idx,
                /* Output Buffers: 1*/
                int* label_ptr, size_t label_size){

    void* root_section = __hetero_section_begin();

    // Re-encode each iteration.
    void* inference_task = __hetero_task_begin(
            /* Input Buffers: 3 */ 8, 
            rp_matrix_ptr, rp_matrix_size, 
            datapoint_vec_ptr, datapoint_vec_size, 
            encoded_hv_ptr, encoded_hv_size,
            classes_ptr, classes_size, 
            label_ptr, label_size,
            scores_ptr, scores_size,
            norms_ptr, norms_size,
            encoded_hv_idx,
            /* Output Buffers: 1 */ 1,encoded_hv_ptr, encoded_hv_size,
            "inference_task"
            );


    __hetero_hdc_inference(
        15,
        (void*) encoding_and_inference_node<D, K, N_VEC, N_FEATURES>,
        rp_matrix_ptr, rp_matrix_size, 
        datapoint_vec_ptr, datapoint_vec_size, 
        classes_ptr, classes_size, 
        label_ptr, label_size,
        encoded_hv_ptr, encoded_hv_size, //Extra Arguments
        scores_ptr, scores_size,
        norms_ptr, norms_size,
        encoded_hv_idx
    );


    __hetero_task_end(inference_task);

    __hetero_section_end(root_section);
}
