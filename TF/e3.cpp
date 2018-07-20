#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <bits/stdc++.h>

using namespace cv;
using namespace std;


//suma y resta //brillo
void brillo(Mat &image, int value){
    Mat new_image = Mat::zeros( image.size(), image.type() );
    for( int y = 0; y < image.rows; y++ ){
        for( int x = 0; x < image.cols; x++ ){
            for( int c = 0; c < 3; c++ ){
                image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( image.at<Vec3b>(y,x)[c] + value );
            }
        }
    }
}




void imprimir(Mat &image){
    int d = image.rows/10;
    for(int c = 0; c < 3; ++c){
        for( int y = 0; y < image.rows; y++ ){
            for( int x = 0; x < image.cols; x++ ){
                int val = image.at<Vec3b>(y,x)[c]; 
                if(y%d==0 && x%d==0){
                    cout<<val<<" ";
                }
            }
            if(y%d==0){
                cout<<endl;
            }
        }
        cout<<"\n"<<endl;
    }
}
/*
void FFT(){
    doubleA[][xydim];
    double freal[][xydim];
    double fimag[][xydim];
    double phi, sum1, sum2;
    for(inti = 0; i <N; i++)
        for(intj = 0; j <N, j++) {
            sum1 = 0;
            sum2 = 0;
            for(intx = 0; x <N; x++)
                for(inty = 0; y <N, y++) {
                    phi= 2 * PI *(i * x + j * y) / N;
                    sum1 = sum1 + A[x][y] * cos(phi);
                    sum2 = sum2 + A[x][y] * sin(phi);
                }

            freal[i][j] = sum1 / N;
            fimag[i][j] = -sum2 / N;
    }
}*/



def DFT2D(image):
    global M, N
    (M, N) = image.size # (imgx, imgy)
    dft2d_red = [M][N] 
    dft2d_grn = [M][N] 
    dft2d_blu = [M][N] 
    pixels = image.load()
    for k in range(M):
        for l in range(N):
            sum_red = 0.0
            sum_grn = 0.0
            sum_blu = 0.0
            for m in range(M):
                for n in range(N):
                    (red, grn, blu, alpha) = pixels[m, n]
                    e = cmath.exp(- 1j * pi2 * (float(k * m) / M + float(l * n) / N))
                    sum_red += red * e
                    sum_grn += grn * e
                    sum_blu += blu * e
            dft2d_red[l][k] = sum_red / M / N
            dft2d_grn[l][k] = sum_grn / M / N
            dft2d_blu[l][k] = sum_blu / M / N
    return (dft2d_red, dft2d_grn, dft2d_blu)






void run(){
    const char* filename = "cab.jpeg";
    Mat I = imread(filename);//, CV_LOAD_IMAGE_GRAYSCALE);
    imshow("original", I );    // Show the result
    
    imprimir(I);



    ////////////////////
    double fps = 60;
    // calcular el tiempo de espera entre cada imagen a mostrar
    int delay = 1000 / fps;
    while (true){
        if(waitKey(delay) == 27) break;
    }

}









int main(){
    run();
    return 0;
}



/*

    if( I.empty())//
        return;

*/