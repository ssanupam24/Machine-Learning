#include <iostream>
#include <mpi.h>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <math.h>
using namespace std;

/*@method to read the data from the file and insert the key value pairs in the 2D array 
 *@param file name,2D array,size of the key-value pairs,number of processors,map to store the mapped keys
 */
void readFromFile(char* filename,int csvData[][2],int size,int nprocs,map<int,int> keyMap){
       ifstream ifs(filename,fstream::in);
       int key,value;
       char c;
       string s;
       int i =0;
       ifs>>s;
       while(ifs>>key>>c>>value){ 
            csvData[i][0] = keyMap[key];
            csvData[i][1] = value;
            i++;
       }
       if(size % nprocs != 0){
            int extra = nprocs - (size % nprocs);
            map<int,int>::iterator it = keyMap.begin();
            for(int j = 0; j < extra; j++){
                csvData[i][0] = it->second;
                csvData[i][1] = 0;
                i++;
            }
       }
}

/*Method to read the file and determine the size of the key value pairs
 *@param file name,size of the key-value pairs,total keys,number of processors,actual size and map for storing the mapped values
 */
void readSize(char* filename,int &size,int &totkeys,int totproc,int& ac_size,map<int,int>& keyMap){
       ifstream ifs(filename,fstream::in);
       int key,value;
       char c;
       string s;
       int s1 = 0;
       int extra = 0;
       totkeys = 0;
       ifs>>s;
       while(ifs>>key>>c>>value){ 
           if(keyMap.find(key) == keyMap.end()){
               keyMap[key] = totkeys;
               totkeys++;
           }
            s1++;
       }
        size = s1;
        ac_size = s1;
        //Adding extra size for making it multiple of number of procs
        if(size % totproc != 0){
            extra = totproc - (size % totproc);
            size += extra;
        }

}

int main(int argc, char* argv[])
{
    char* file_name = "100000_key-value_pairs.csv";
    int size = 0;
    int totkeys = 0;
    int j = 0;
    int ac_size = 0;
    int rank,proc_size;
    map<int,int> keyMap;
    ofstream file("Output_Task2.txt");
    MPI_Init(&argc,&argv);
    MPI_Comm_size (MPI_COMM_WORLD, &proc_size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    //Determine the size of the file and store the mapped values as well
    if(rank == 0){
    readSize(file_name,size,totkeys,proc_size,ac_size,keyMap);
    }
    MPI_Bcast(&size,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&totkeys,1,MPI_INT,0,MPI_COMM_WORLD);
    int data[size][2];
    //Read the key and value pairs from the file and store it in the map according to the mapped values
    if(rank == 0){
    readFromFile(file_name, &data[0],ac_size,proc_size,keyMap);
    }
    int sub_size = size/proc_size;
    int sub_list[size][2];
    //Scatter the data and send the data values to the processors
    MPI_Scatter(data,sub_size*2,MPI_INT,sub_list,sub_size*2,MPI_INT,0,MPI_COMM_WORLD);
    //cout<<"Scatter done"<<endl;
    int sub_list1[totkeys];
    int sub_list2[proc_size][totkeys];
    memset(sub_list1,0,totkeys*sizeof(int));
    //Here I am trying to group the values corresponding to the keys
    for(int i =0; i < sub_size; i++){
         sub_list1[sub_list[i][0]] = sub_list1[sub_list[i][0]] +   sub_list[i][1];
    }
    //Need to gather all the values computed by all the processes
    MPI_Allgather(sub_list1, totkeys, MPI_INT, sub_list2, totkeys, MPI_INT, MPI_COMM_WORLD);
    //cout<<"Gather done"<<endl;
    int size2 = ceil((float)totkeys/(float)proc_size);
    int sub_list3[size2];
    memset(sub_list3,0,sizeof(int)*size2);
    int final_list[totkeys];
    int start = size2*rank;
    int end = size2*(rank+1);
    if(end > totkeys)
        end = totkeys;
    for(int i =start,j=0;i < end;i++,j++){
        for(int k =0;k < proc_size;k++){
            sub_list3[j] += sub_list2[k][i];
        }
    }
    int count1[proc_size],offset[proc_size];
    for(int i =0; i < proc_size; i++){
        int s = size2*i;
        int e = size2*(i+1);
        if(e > totkeys)
            e = totkeys;
        count1[i] = e-s;
        if(i == 0)
            offset[i]=0;
        else
            offset[i]=offset[i-1] + count1[i-1];
    }
    //The blocks of key value pairs are gathered in one single master processor.
    MPI_Gatherv(&sub_list3, end-start, MPI_INT, &final_list,count1,offset, MPI_INT, 0, MPI_COMM_WORLD);
    //cout<<"combining done"<<endl;
    if(rank == 0){
        map<int,int>::iterator it;
        for(it= keyMap.begin();it!=keyMap.end();it++){
            file<<it->first<<" "<<final_list[it->second]<<endl;
        }
    }
    MPI_Finalize ();
    return 0;
}
