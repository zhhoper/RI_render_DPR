/*This file use libigl as rigid as possible to deform triangle
 * input: 
 *        1. a file containing the triangle 
 *        2. a file containing the fixed point index and its destination
 *        3. width of an image
 *        4. height of an image
 * outpout:
 *        a file that contains the obj file of deformation
 *        */

#include <iostream>
#include <fstream>
#include <list>
#include <igl/arap.h>
#include <stdlib.h>

std::vector<std::string> string_split(std::string s){
    // split a string by space
    std::string delimiter = " ";
    size_t pos = 0;
    std::string token;
    std::vector<std::string> out;
    while ((pos = s.find(delimiter)) != std::string::npos) {
            token = s.substr(0, pos);
            s.erase(0, pos + delimiter.length());
            out.push_back(token);
    }
    if(s.size() != 0){
        out.push_back(s);
    }
    return out;
}

int load_triangle(char *srcFile, Eigen::MatrixXd &V, Eigen::MatrixXi &F){
    // load triangle from a file
    std::list<double> tmp_vertex;
    std::list<int> tmp_face;
    std::ifstream myFile;
    myFile.open(srcFile);
    while( !myFile.eof()) {
        std::string tmp;
        std::getline(myFile, tmp);
        std::vector<std::string> out = string_split(tmp);
        if(out.size() != 0){
            if (out[0].compare("v") == 0){
                for (int i = 1; i < out.size(); i++){
                    tmp_vertex.push_back(std::stod(out[i]));
                }
            }
            if (out[0].compare("f") == 0){
                for (int i = 1; i < out.size(); i++)
                    tmp_face.push_back(std::stoi(out[i])-1);
            }
        }        
    }
    myFile.close();
    // create eigen matrix
    int numVertex = 2;
    int numFace = 3;
    int count = 0;
    V.resize(int(tmp_vertex.size()/numVertex), numVertex);
    for (auto const&i : tmp_vertex){
        int row = count/numVertex;
        int col = count%numVertex;
        V(row, col)=i;
        count += 1;
    }
    count = 0;
    F.resize(int(tmp_face.size()/numFace), numFace);
    for (auto const&i : tmp_face){
        int row = count/numFace;
        int col = count%numFace;
        F(row, col)=i;
        count += 1;
    }
    return 0;
}

int load_index(char *srcFile, Eigen::VectorXi &b, Eigen::MatrixXd &bc){ 
    // load fixed points and its position
    std::list<int> tmp_index;
    std::list<double> tmp_vertex;
    std::ifstream myFile;
    myFile.open(srcFile);
    while( !myFile.eof()) {
        std::string tmp;
        std::getline(myFile, tmp);
        std::vector<std::string> out = string_split(tmp);
        if(out.size() == 3){
            tmp_index.push_back(std::stoi(out[0]));
            tmp_vertex.push_back(std::stod(out[1]));
            tmp_vertex.push_back(std::stod(out[2]));
        }        
    }

    int numVertex = 2;
    // create b
    b.resize(tmp_index.size());
    int count = 0;
    for (auto const&i : tmp_index){
        b[count] = i;
        count += 1;
    }
    // create bc
    bc.resize(int(tmp_vertex.size()/numVertex), numVertex);
    count = 0;
    for (auto const&i : tmp_vertex){
        int row = count/numVertex;
        int col = count%numVertex;
        bc(row, col) = i;
        count += 1;
    }
    return 0;
}

int saveObjFile(char *saveName, Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &U, float width, float height){
    // save deformed mesh as obj file
    std::ofstream myfile(saveName);
    
    // save vertices
    for (int i = 0; i < U.rows(); i++){
        myfile<<"v ";
        for (int j = 0; j < U.cols(); j++){
            myfile<<U(i,j)<<" ";
        }
        myfile<<0<<std::endl;
    }

    // save texture
    for (int i = 0; i < V.rows(); i++){
        myfile<<"vt ";
        myfile<<V(i,0)/(width-1)<<" ";
        myfile<<V(i,1)/(height-1)<<" ";
        //for (int j = 0; j < V.cols(); j++){
        //    myfile<<V(i,j)<<" ";
        //}
        myfile<<0<<std::endl;
    }
    // save face
    for (int i = 0; i < F.rows(); i++){
        myfile<<"f ";
        for (int j=0; j<F.cols(); j++){
            myfile<<F(i,j)+1<<" ";
        }
        myfile<<std::endl;
    }
    myfile.close();
    return 0;
}

int main(int argc, char** argv){
    /*arguments:
    *        1. a file containing the triangle 
    *        2. a file containing the fixed point index and its destination
    *        3. save file (obj file that saves the deformation)
    *        3. width of an image
    *        4. height of an image
    *        */
    if (argc != 6) {
        std::cout<<"number of input should be 6"<<std::endl;
        return 0;
    }
    Eigen::MatrixXd V;     // vertices of triangle
    Eigen::MatrixXi F;     // faces of a triangle
    Eigen::MatrixXd bc;    // target position of the control points
    Eigen::VectorXi b;     // index of control points
    // load V and F
    load_triangle(argv[1], V, F);
    //std::cout<<"V size "<<V.size()<<std::endl;
    //std::cout<<"F size "<<F.size()<<std::endl;
    // load bc and b
    load_index(argv[2], b, bc);
    //std::cout<<"b size "<<b.size()<<std::endl;
    //std::cout<<"bc size "<<bc.size()<<std::endl;
    // get width and height of the image
    float width = atof(argv[4]);
    float height = atof(argv[5]);
    //std::cout<<"width of the image "<<width<<std::endl;
    //std::cout<<"height of the image "<<height<<std::endl;

    // deform
    igl::ARAPData arap_data;
    arap_data.max_iter = 100;
    igl::arap_precomputation(V,F,V.cols(),b,arap_data);
    Eigen::MatrixXd U(V);
    //std::cout<<"U size "<<U.size()<<std::endl;
    igl::arap_solve(bc,arap_data,U);

    // save file
    //std::cout<<"checkpoint 4"<<std::endl;
    saveObjFile(argv[3], V, F, U, width, height);
    std::cout<<"save as "<<argv[3]<<" obj file"<<std::endl;
    //std::cout<<"checkpoint 5"<<std::endl;

    return 0;
}
