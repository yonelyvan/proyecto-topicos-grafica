//`pkg-config opencv --cflags --libs`
//#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <bits/stdc++.h>

using namespace cv;
using namespace std;


void print_img_test(){
    Mat image = imread("img.jpg", CV_LOAD_IMAGE_COLOR);
    imshow( "img", image ); 
    waitKey(0);
}

Mat brillo(Mat &image, int value){
    Mat new_image = Mat::zeros( image.size(), image.type() );
    for( int y = 0; y < image.rows; y++ ){
        for( int x = 0; x < image.cols; x++ ){
            for( int c = 0; c < 3; c++ ){
                new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( image.at<Vec3b>(y,x)[c] + value );
            }
        }
    }
   return new_image;
}


//'kernel'
void add_brillo_kernel(int *M, int value, int rows, int cols){
    for (int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            M[cols * i + j] += value;
        }
    }
}


Mat brillo_cuda(Mat &image, int value){
    int rows = image.rows;
    int cols = image.cols;
    int nElem = rows * cols;
    int * R = (int *) malloc(nElem * sizeof(int));
    int * G = (int *) malloc(nElem * sizeof(int));
    int * B = (int *) malloc(nElem * sizeof(int));

    //load IMG
    for( int i = 0; i < image.rows; i++ ){
        for( int j = 0; j < image.cols; j++ ){
            R[cols*i+j] = image.at<Vec3b>(i,j)[0];
            G[cols*i+j] = image.at<Vec3b>(i,j)[1];
            B[cols*i+j] = image.at<Vec3b>(i,j)[2];

        }
    }
    add_brillo_kernel(R,100,rows,cols);
    add_brillo_kernel(G,100,rows,cols);
    add_brillo_kernel(B,100,rows,cols);

    Mat new_image = Mat::zeros( image.size(), image.type() );
    for( int i = 0; i < image.rows; i++ ){
        for( int j = 0; j < image.cols; j++ ){
            new_image.at<Vec3b>(i,j)[0] = saturate_cast<uchar>( R[cols*i+j] );
            new_image.at<Vec3b>(i,j)[1] = saturate_cast<uchar>( G[cols*i+j] );
            new_image.at<Vec3b>(i,j)[2] = saturate_cast<uchar>( B[cols*i+j] );
        }
    }
    return new_image;
}



void aumentar_brillo(){
    Mat img = imread("img.jpg", CV_LOAD_IMAGE_COLOR);
    imshow( "original", img ); 
    
    Mat img2 =brillo(img,100);
    imshow( "brillo", img2 ); 
    waitKey(0);
}



int main( int argc, char** argv ){
    //print_img_test();       
    aumentar_brillo();
    return 0;
}