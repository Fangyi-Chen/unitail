
#include "poly_overlaps.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdio>
#include<algorithm>

using namespace std;

//##define CUDA_CHECK(condition)\
//
//  do {
//    cudaError_t error = condition;
//    if (error != cudaSuccess) {
//
//    }
//  }

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
// int const threadsPerBlock = sizeof(unsigned long long) * 8;


#define maxn 510
const double eps=1E-8;

__device__ inline int sig(float d){
    return(d>eps)-(d<-eps);
}
// struct Point{
//     double x,y; Point(){}
//     Point(double x,double y):x(x),y(y){}
//     bool operator==(const Point&p)const{
//         return sig(x-p.x)==0&&sig(y-p.y)==0;
//     }
// };

__device__ inline int point_eq(const float2 a, const float2 b) {
    return (sig(a.x - b.x) == 0) && (sig(a.y - b.y)==0);
}

__device__ inline void point_swap(float2 *a, float2 *b) {
    float2 temp = *a;
    *a = *b;
    *b = temp;
}

__device__ inline void point_reverse(float2 *first, float2* last)
{
    while ((first!=last)&&(first!=--last)) {
        point_swap (first,last);
        ++first;
    }
}

__device__ inline float cross(float2 o,float2 a,float2 b){  //叉积
    return(a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);
}
__device__ inline float area(float2* ps,int n){
    ps[n]=ps[0];
    float res=0;
    for(int i=0;i<n;i++){
        res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
    }
    return res/2.0;
}
__device__ inline int lineCross(float2 a,float2 b,float2 c,float2 d,float2&p){
    float s1,s2;
    s1=cross(a,b,c);
    s2=cross(a,b,d);
    if(sig(s1)==0&&sig(s2)==0) return 2;
    if(sig(s2-s1)==0) return 0;
    p.x=(c.x*s2-d.x*s1)/(s2-s1);
    p.y=(c.y*s2-d.y*s1)/(s2-s1);
    return 1;
}

__device__ inline void polygon_cut(float2*p,int&n,float2 a,float2 b, float2* pp){
    // TODO: The static variable may be the reason, why single thread is ok, multiple threads are not work
    // printf("polygon_cut, offset\n");
    
    // static float2 pp[maxn];
    int m=0;p[n]=p[0];
    for(int i=0;i<n;i++){
        if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
        if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
            lineCross(a,b,p[i],p[i+1],pp[m++]);
    }
    n=0;

    for(int i=0;i<m;i++)
        if(!i||!(point_eq(pp[i], pp[i-1])))
            p[n++]=pp[i];
    // while(n>1&&p[n-1]==p[0])n--;
    while(n>1&&point_eq(p[n-1], p[0]))n--;
    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // // corresponding to k
    // int y = blockIdx.y * blockDim.y + threadIdx.y;
    // int offset = x * 1 + y;
    // printf("polygon_cut, offset\n");
}

//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
__device__ inline float intersectArea(float2 a,float2 b,float2 c,float2 d){
    float2 o = make_float2(0,0);
    int s1=sig(cross(o,a,b));
    int s2=sig(cross(o,c,d));
    if(s1==0||s2==0)return 0.0;//退化，面积为0
    if(s1 == -1) point_swap(&a, &b);
    if(s2 == -1) point_swap(&c, &d);
    float2 p[10]={o,a,b};
    int n=3;
    float2 pp[maxn];
    polygon_cut(p,n,o,c,pp);
    polygon_cut(p,n,c,d,pp);
    polygon_cut(p,n,d,o,pp);
    float res=fabs(area(p,n));
    if(s1*s2==-1) res=-res;return res;

}
//求两多边形的交面积
// TODO: here changed the input, this need to be debug
__device__ inline float intersectArea(float2*ps1,int n1,float2*ps2,int n2){
    if(area(ps1,n1)<0) point_reverse(ps1,ps1+n1);
    if(area(ps2,n2)<0) point_reverse(ps2,ps2+n2);
    ps1[n1]=ps1[0];
    ps2[n2]=ps2[0];
    float res=0;
    for(int i=0;i<n1;i++){
        for(int j=0;j<n2;j++){
            res+=intersectArea(ps1[i],ps1[i+1],ps2[j],ps2[j+1]);
        }
    }
    return res;//assumeresispositive!
}

__device__ inline void RotBox2Poly(float const * const dbox, float2 * ps) {
    float cs = cos(dbox[4]);
    float ss = sin(dbox[4]);
    float w = dbox[2];
    float h = dbox[3];

    float x_ctr = dbox[0];
    float y_ctr = dbox[1];
    ps[0].x = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0);
    ps[1].x = x_ctr + cs * (w / 2.0) - ss * (h / 2.0);
    ps[2].x = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0);
    ps[3].x = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0);

    ps[0].y = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0);
    ps[1].y = y_ctr + ss * (w / 2.0) + cs * (h / 2.0);
    ps[2].y = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0);
    ps[3].y = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0);
}


__device__ inline float devPolyIoU(float const * const p, float const * const q) {
    float2 ps1[maxn], ps2[maxn];
    int n1 = 4;
    int n2 = 4;
    for (int i = 0; i < 4; i++) {
        ps1[i].x = p[i * 2];
        ps1[i].y = p[i * 2 + 1];

        ps2[i].x = q[i * 2];
        ps2[i].y = q[i * 2 + 1];
    }
    float inter_area = intersectArea(ps1, n1, ps2, n2);
    float union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    float iou = 0;
    if (union_area == 0) {
        iou = (inter_area + 1) / (union_area + 1);
    } else {
        iou = inter_area / union_area;
    }
    return iou;

}

__global__ void overlaps_kernel(const int N, const int K, const float* dev_boxes,
                           const float * dev_query_boxes, float* dev_overlaps) {
  
  // corresponding to n
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  // corresponding to k
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < N) && (y < K)) {
    int offset = x * K + y;
    dev_overlaps[offset] = devPolyIoU(dev_boxes + x * 8, dev_query_boxes + y * 8);
  } 
}

// __global__ void overlaps_kernel(const int N, const int K, const float* dev_boxes,
//     const float * dev_query_boxes, float* dev_overlaps) {
//         printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
// }


void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}


void _overlaps(float* overlaps, const float* boxes,const float* query_boxes, int n, int k, int device_id) {

  _set_device(device_id);

  float* overlaps_dev = NULL;
  float* boxes_dev = NULL;
  float* query_boxes_dev = NULL;

  CUDA_CHECK(cudaMalloc((void**)&boxes_dev, n * 8 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes,
                        n * 8 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc((void**)&query_boxes_dev, k * 8 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(query_boxes_dev,
                        query_boxes,
                        k * 8 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc((void**)&overlaps_dev, n * k * sizeof(float)));

  dim3 blocks(DIVUP(n, 32),
              DIVUP(k, 32));
  dim3 threads(32, 32);

  overlaps_kernel<<<blocks, threads>>>(n, k, boxes_dev, query_boxes_dev, overlaps_dev);
  CUDA_CHECK(cudaMemcpy(overlaps,
                        overlaps_dev,
                        n * k * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(overlaps_dev));
  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(query_boxes_dev));

}
