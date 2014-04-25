#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <iomanip>
#include <omp.h>
#include <stdlib.h>
#include <utility>
#include <math.h>
using namespace std;

/*Method to read the data from the file and insert the nodes into matrix and also determine the size of the graph
 *@param file name,2D matrix and outlink vector
 */
void readFromFile(char* filename,vector<vector<pair<int,double> > >& matrix,vector<int>& outLinks){
       ifstream ifs(filename,fstream::in);
       setprecision(20);
       int row,col;
       int i =0;
       while(ifs>>col>>row){ 
            matrix[row].push_back(make_pair(col,1.0));
            matrix[col].push_back(make_pair(row,1.0));
            outLinks[col] += 1;
            outLinks[row] += 1;
       }

}
/*Method to read the file and determine the size of the nodes
 *@param file name and size variable
 */
void readSize(char* filename,int &size){
            ifstream ifs(filename,fstream::in);
            setprecision(20);
            int row,col;
            set<int> s1;
            while(ifs>>col>>row){
                s1.insert(row);
                s1.insert(col);
            }
            size = s1.size();
}
/*Method to normalize the rank vector
 *@param total number of nodes
 */
void normalize(int size,vector<vector<pair<int,double> > >& matrix,vector<int> outLinks,vector<double>& vect){
    setprecision(20);
#pragma omp parallel for
    for(int i =0; i < size; i++){
        vect[i]=(double)(1.0/size);
        for(int j =0; j < matrix[i].size(); j++){
            if(outLinks[j] != 0){
                matrix[i][j].second = (double)(matrix[i][j].second/outLinks[matrix[i][j].first]);
                }
            }
        }
     }
/*Main function to execute the iterations and calculate the page ranks
 *
 */

int main(){
    omp_set_num_threads(4);
    int size = 0;
    typedef pair<int,double> pairs;
    double start_time,run_time;
    start_time = omp_get_wtime();
    const double d = 0.85;
    char* file_name = "facebook_combined.txt";
    ofstream out_file("Output_Task1.txt");
    setprecision(20);
    readSize(file_name,size);
    vector<vector<pairs> > matrix(size);
    vector<int> outLinks(size);
    vector<double> vect(size);
    readFromFile(file_name,matrix,outLinks);
    vector<double> prevRank(size);
    normalize(size,matrix,outLinks,vect);
    double epsilon = 0.0000000001;
    int iter = 0;
    while(1){
#pragma omp parallel for
        for(int i =0; i < size; i++){
            prevRank[i] = vect[i];
            vect[i] = 0.0;
        }
        //make it parallel
#pragma omp parallel for collapse(1) schedule(static)
        for(int i =0; i < size; i++){
            for(int j =0; j < matrix[i].size(); j++){
                vect[i] +=  matrix[i][j].second*prevRank[matrix[i][j].first];
              }   
            vect[i] = (double)(d*vect[i]) + (double)(1.0-d)/size;
        }
        iter++;
        int count = 0;
#pragma omp parallel for
        for(int i =0; i < size; i++){
            if(fabs(prevRank[i] - vect[i]) < epsilon)
#pragma omp atomic
                count++;
        }
        if(count == size){
            cout<<"Converged"<<endl;
            break;
        }
    }
    run_time = omp_get_wtime() - start_time;
    cout<<"Runtime"<<run_time<<endl;

    for(int i =0; i < size; i++){
         out_file<<vect[i]<<endl;
          }
    cout<<"Total Iterations are "<<iter<<endl;
}
