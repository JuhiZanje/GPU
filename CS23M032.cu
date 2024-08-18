#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here

__device__ int gcm(int a,int b){
    if(b==0)    return a;
    return gcm(b,a%b);
}

__global__ void setDHTempScore(int *DHP,int *tempDHP,int *Dscore,int H){
    int id=threadIdx.x;
    DHP[id]=H;
    tempDHP[id]=H;
    Dscore[id]=0;
}

__global__ void initGCM(int *gcmStore,int T){
    int ind=blockIdx.x * T + threadIdx.x;
    gcmStore[ind]=0;
}

__global__ void updateDH(int *DHP,int *tempDHP, int *countActiveTank,int T,int round){
    if((round%T)!=0){
        int id=threadIdx.x;
        if(tempDHP[id]<=0 && DHP[id]>0)
            atomicAdd(countActiveTank,-1);        
        DHP[id]=tempDHP[id];
    }                    
}

__global__ void roundKernel(int round,int T,int *DHP,int *tempDHP,int* Dxcoord,int* Dycoord,int* Dscore,int *countActiveTank,int *gcmStore){
    __shared__ int tankToShoot;
    __shared__ volatile int lock;
    __shared__ volatile int minK;
    if((round%T)!=0){
        int j=threadIdx.x;
        int id=blockIdx.x;

        if(j==0){
            minK=INT_MAX;
            tankToShoot=-1;
            lock=0;
        }

        __syncthreads();

        if(DHP[id]>0){
            int hitDir=(id+round)%T;
            int x=Dxcoord[id],y=Dycoord[id];
            int diffx=Dxcoord[hitDir]-x;
            int diffy=Dycoord[hitDir]-y;
            int gcmVal,ind;
            ind=hitDir * T + id;
            if(gcmStore[ind]!=0){
                gcmVal=gcmStore[ind];
            }else{
                gcmVal=abs(gcm(diffx,diffy));
                gcmStore[ind]=gcmVal;
            }        
            int dirx=diffx/gcmVal;
            int diry=diffy/gcmVal;

            if(DHP[j]>0 && id!=j ){
                diffx=Dxcoord[j]-x;
                diffy=Dycoord[j]-y;
                ind=j * T + id;
                if(gcmStore[ind]!=0){
                    gcmVal=gcmStore[ind];
                }else{
                    gcmVal=abs(gcm(diffx,diffy)); 
                    gcmStore[ind]=gcmVal;
                }                

                int dirxN=diffx/gcmVal;
                int diryN=diffy/gcmVal;                

                if(dirxN==dirx && diry==diryN){
                    int k;
                    if(dirx!=0)
                        k=diffx/dirx;
                    else
                        k=diffy/diry;
                    int old;
                    if(k<minK){
                        for(int i=0;i<32;i++){
                            if(j%32==i){
                                do{
                                    old=atomicCAS((int *)&lock,0,1);
                                    if(old==0){
                                        if(k<minK){ 
                                            minK=k;
                                            tankToShoot=j;
                                        }
                                        lock=0;
                                    }
                                }while(old!=0);
                            }
                        }               
                    
                    }                
                }
            }            

        }
        __syncthreads();    
        if(j==0){
            if(tankToShoot!=-1){            
                atomicAdd(&tempDHP[tankToShoot],-1);            
                Dscore[id]++;
            }
        } 
    }
         
}


//***********************************************


int main(int argc,char **argv)
{    
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int id=0;id<T;id++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[id] );
      fscanf( inputfilepointer, "%d", &ycoord[id] );
    }
		

    auto start = chrono::high_resolution_clock::now();


    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int *countActiveTank;
    int *Dxcoord,*Dycoord,*Dscore;
    cudaMalloc(&Dxcoord , T * sizeof (int)) ;
	cudaMemcpy(Dxcoord,xcoord,T * sizeof (int),cudaMemcpyHostToDevice);

    cudaMalloc(&Dycoord , T * sizeof (int)) ;
	cudaMemcpy(Dycoord,ycoord,T * sizeof (int),cudaMemcpyHostToDevice);

    cudaMalloc(&Dscore , T * sizeof (int)) ;
	// cudaMemcpy(Dscore,score,T * sizeof (int),cudaMemcpyHostToDevice);

    int *DH;
    cudaMalloc(&DH , T * sizeof (int));

    int *tempDH;
    cudaMalloc(&tempDH , T * sizeof (int));
    setDHTempScore<<<1,T>>>(DH,tempDH,Dscore,H);

    int *gcmStore;
    cudaMalloc(&gcmStore , T * T* sizeof (int));   
    initGCM<<<T,T>>>(gcmStore,T);    

    cudaHostAlloc(&countActiveTank,sizeof(int),0);
    *countActiveTank=T;
    int round=1; 
    
    while(*countActiveTank>1){  
        roundKernel<<<T,T>>>(round,T,DH,tempDH,Dxcoord,Dycoord,Dscore,countActiveTank,gcmStore);
        updateDH<<<1,T>>>(DH,tempDH,countActiveTank,T,round);              
        cudaDeviceSynchronize();        
        round++;       
    }   

	cudaMemcpy(score,Dscore,T * sizeof (int),cudaMemcpyDeviceToHost);   
    
    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int id=0;id<T;id++)
    {
        fprintf( outputfilepointer, "%d\n", score[id]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}