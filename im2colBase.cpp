#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
extern"C"
{
    #include<cblas.h>
}
using namespace std;

const int KERNEL_NUM = 5;
const int KERNEL_H = 11;
const int KERNEL_W = 11;
const int IMG_H = 227;
const int IMG_W = 227;

void im2col_cpu(float** data_im, const int height,
        const int width, const int kernel_h,
        const int kernel_w, float* data_col,
        const int stride=1);

int main() {
    // 初始化kernel_list矩阵
    float** kernel_list[KERNEL_NUM];
    for (int i=0;i<KERNEL_NUM;i++) {
        kernel_list[i] = new float*[KERNEL_H];
        for (int j=0;j<KERNEL_H;j++) {
            kernel_list[i][j] = new float[KERNEL_W];
            for (int k=0; k<KERNEL_W;k++) {
                kernel_list[i][j][k] = rand() % 9 + 1;
            }
        }
    }
    // 初始化输入矩阵
    float** img = new float*[IMG_H];
    for (int i = 0; i < IMG_H; i++) {
        img[i] = new float[IMG_W];
        for (int j = 0; j < IMG_W; j++) {
            img[i][j] = rand() % 9 + 1;
        }
    }

    // 开始计时
    struct timeval tstart, tend;
    gettimeofday(&tstart, NULL);

    // 展开kernel_list矩阵
    float* kernel2col = new float [KERNEL_NUM*KERNEL_H*KERNEL_W];
    for (int i = 0, count=0; i < KERNEL_NUM; i++) {
        for (int j=0;j<KERNEL_H; j++) {
            for (int k=0;k<KERNEL_W;k++,count++) {
                kernel2col[count] = kernel_list[i][j][k];
            }
        }
    }

    // 展开输入矩阵
    int output_w = IMG_W-KERNEL_W + 1;
    int output_h = IMG_H-KERNEL_H + 1;
    float* img2col = new float[KERNEL_H*KERNEL_W*output_h*output_w];
    im2col_cpu(img, IMG_H, IMG_W, KERNEL_H, KERNEL_W, img2col);

    // 使用BLAS进行大矩阵乘法运算
    float* filter_img = new float[KERNEL_NUM*output_h*output_w];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,KERNEL_NUM,
        output_h*output_w,KERNEL_W*KERNEL_H,1,
        kernel2col,KERNEL_H*KERNEL_W,
        img2col,output_h*output_w,0,filter_img,output_h*output_w);

    // 释放内存
    delete []kernel2col;
    delete []img2col;
    delete []filter_img;

    // 结束计时
    gettimeofday(&tend, NULL);
    cout<<"Total time cost: "<<(tend.tv_sec-tstart.tv_sec)*1000 + (tend.tv_usec-tstart.tv_usec)/1000<<" ms"<<endl;

    // 释放kernel_list矩阵所占内存
    for (int i=0;i<KERNEL_NUM;i++) {
        for (int j=0;j<KERNEL_H;j++) {
            delete kernel_list[i][j];
        }
        delete kernel_list[i];
    }

    // 释放输入矩阵所占内存
    for (int i = 0; i < IMG_H; i++) {
        delete img[i];
    }
}

void im2col_cpu(float** data_im, const int height,
    const int width, const int kernel_h,
    const int kernel_w, float* data_col,
    const int stride) {
    const int output_w = width - (kernel_w-1);
    const int output_h = height - (kernel_h - 1);
    for (int kernel_row = 0,count = 0;  kernel_row < kernel_h; kernel_row++){
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
            int kernel_index = kernel_row*kernel_w+kernel_col;
            // 获取所有滑块在[kernel_row,kernel_col]位置的值，存储到展开矩阵的第kernel_row*kernel_w+kernel_col-1行
            int input_row = kernel_row;
            for (int output_rows = output_h; output_rows; output_rows--) {
                int input_col = kernel_col;
                // 完成当前行每个滑块[kernel_row，kernel_col]位置的元素的提取
                for (int output_col = output_w; output_col; output_col--,count++) {
                    // 完成当前行,当前列的滑块的[kernel_row,kernel_col]位置元素的提取
                    // cout << "replace data_col "<< *(data_col[kernel_index]) <<" with: "<<data_im[input_row][input_col] << endl;
                    data_col[count] = data_im[input_row][input_col];
                    // cout << "after replaced"<< *(data_col[kernel_index]+count) << endl;
                    input_col += stride;
                }
                input_row += stride;
            }
        }
    }
}
