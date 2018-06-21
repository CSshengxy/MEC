#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
extern"C"
{
    #include<cblas.h>
}
#include <omp.h>
using namespace std;

const int KERNEL_NUM = 5;
const int KERNEL_H = 11;
const int KERNEL_W = 11;
const int IMG_H = 1000;
const int IMG_W = 1000;
const int THREAD_NUM = 4;

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
    float* img2col = new float[output_w*IMG_H*KERNEL_W];
    im2col_cpu(img, IMG_H, IMG_W, KERNEL_H, KERNEL_W, img2col);

    // cout << "输入矩阵-------------------------" << endl;
    // for (int i = 0; i < IMG_H; i++) {
    //     for (int j = 0; j < IMG_W; j++) {
    //         cout << img[i][j] << " ";
    //         if (j == IMG_W - 1)
    //             cout << endl;
    //     }
    // }
    // cout << "--------------------------------" << endl;
    // cout << "输入矩阵展开----------------------" << endl;
    // for (int i = 0; i < IMG_H*KERNEL_W; i++) {
    //     for (int j = 0; j < output_w; j++) {
    //         cout << img2col[i*output_w+j] << " ";
    //         if (j == output_w - 1)
    //             cout << endl;
    //     }
    // }
    // cout << "--------------------------------" << endl;

    // TODO:使用BLAS进行大矩阵乘法运算
    float** filter_img_list = new float*[output_h];
    #pragma omp parallel for num_threads(THREAD_NUM)
    for (int i=0; i < output_h ; i++) {
        filter_img_list[i] = new float[KERNEL_NUM*output_w];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,KERNEL_NUM,
            output_w,KERNEL_W*KERNEL_H,1,
            kernel2col,KERNEL_H*KERNEL_W,
            img2col+i*output_w,output_w,0,filter_img_list[i],output_w);
    }
    // cout << "结果矩阵----------------------" << endl;
    // for (int i = 0; i < output_h; i++) {
    //     for (int j = 0; j < KERNEL_NUM; j++) {
    //         for (int k = 0; k < output_w; k++) {
    //             cout << filter_img_list[i][j*output_w+k] << " ";
    //             if (k == output_w - 1)
    //                 cout << endl;
    //         }
    //         if (j == KERNEL_NUM - 1)
    //             cout << endl;
    //     }
    // }
    // cout << "--------------------------------" << endl;

    // 释放内存
    delete []kernel2col;
    delete []img2col;
    for (int i=0; i < output_h ; i++)
        delete []filter_img_list[i];
    delete []filter_img_list;

    // 结束计时
    gettimeofday(&tend, NULL);
    cout<<"[im2colOpt " << IMG_H << "*" << IMG_W << "]Total time cost: "<<(tend.tv_sec-tstart.tv_sec)*1000 + (tend.tv_usec-tstart.tv_usec)/1000<<" ms"<<endl;

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
    #pragma omp parallel for num_threads(THREAD_NUM)
    for (int output_col = 0; output_col < output_w; output_col++) {
        int output_row = 0;
        for (int input_row = 0; input_row < height; input_row++) {
            for (int input_col = output_col; input_col < output_col + kernel_w; input_col++) {
                data_col[output_row*output_w+output_col] = data_im[input_row][input_col];
                output_row++;
            }
        }
    }
}
