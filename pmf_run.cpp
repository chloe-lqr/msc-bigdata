#include <iostream>
#include <sstream>
#include <string.h>
#include <string>
#include <vector>
#include <fstream>
#include <stdio.h>
//#include <stdlib.h>
#include <math.h>
#include <random>
//#include <algorithm>
#include <set>
#include <map>

using namespace std;

class Matrix
{
public:
    int row;
    int column;
    map<string, vector<float>> matrix;

    Matrix() {};
    // create a matrix with fixed size
    Matrix(int row_,int column_,const set<string> sets)
    {
        this->row=row_;
        this->column=column_;
        set<std::string>::iterator it;
        vector<float> dimensions(column);
        for(it=sets.begin(); it!=sets.end(); it++)
        {
            matrix.insert(std::make_pair(*it, dimensions));
        }
    };

    void print()
    {
        map<string, vector<float>>::iterator iter;
        for(iter = matrix.begin(); iter != matrix.end(); iter++)
        {
            for(int i=0; i<iter->second.size(); i++)
            {
                std::cout<<iter->second[i]<<' ';
            }
            std::cout<<'\n';
        }
    };

};

void loadFile(Matrix& M, string filename);
void preprocess(string filename, set<string>& user, set<string>& item, map<string, map<string,float>>&  rating, float maxRating, int& recordNum);
void initUVMatrix(Matrix& M, float mean, float variance);
void sgd(int train_num, int maxRating, Matrix& U, Matrix& V, int dim, float momentum, float u_lambda, float v_lambda, string filename);
void predict(string testFilename, string outputFilename, Matrix U, Matrix V, int recordNum, int dim);

int main()
{
    // preprocess train data, figure out user sets, item sets and ratings.
    std::string trainFilename = "D://bigdata/train.dat";
    std::string testFilename = "D://bigdata/test.dat";
    std::string outputFilename = "D://bigdata/result.txt";
    set<string> user;
    set<string> item;
    map<string, map<string,float>> rating;
    float maxRating = 5.0;
    int recordNum = 0;
    preprocess(trainFilename, user, item, rating, maxRating, recordNum);

    // create U and V, initialize U/V with gauss_distribution
    int dim = 9; // Number of latent features
    float user_mean = 0;
    float user_variance = 0.1;
    float item_mean = 0;
    float item_variance = 0.1;
    Matrix U(user.size(), dim, user);
    initUVMatrix(U, user_mean, user_variance);
    Matrix V(item.size(), dim, item);
    initUVMatrix(V, item_mean, item_variance);

    int maxEpoch = 100; // Number of epoch before stop
    int currentEpoch = 0;
    int batch_size=3;  // Number of training samples used in each batches
    int num_batches=recordNum/batch_size; // Number of batches in each epoch
    int train_num = 100; // the number of iteration
    float momentum = 0.8; // momentum of the gradient
    float u_lambda = 0.1;
    float v_lambda = 0.1;

    sgd(train_num, maxRating, U, V, dim, momentum, u_lambda, v_lambda, trainFilename);
    //U.print();
    //V.print();

    // minibatchProcess();

    predict(testFilename, outputFilename, U, V, recordNum,dim);
}


void preprocess(string filename, set<string>& user, set<string>& item, map<string, map<string,float>>& rating, float maxRating, int& recordNum)
{
    std::ifstream filestream(filename.c_str());
    std::string line;
    while (std::getline(filestream, line))
    {
        vector<std::string> record;
        char *p;
        char *s =(char*)line.data();
        const char *delim = "\t";
        p = strtok(s, delim);
        user.insert(p);
        string userTmp = p;
        p = strtok(NULL, delim);
        item.insert(p);
        string itemTmp = p;
        p = strtok(NULL, delim);
        //float rankingTmp = std::stod(p);
        istringstream istr(p);
        float rankingTmp;
        istr>>rankingTmp;

        // normalize to [0,1] to improve stability
        rankingTmp = (rankingTmp - 1) / (maxRating - 1);

        map<string, map<string,float>>::iterator iter;
        map<string, float> itemRantingTmp;
        iter = rating.find(userTmp);
        // user does not exist in rating matrix
        if(iter == rating.end())
        {
            itemRantingTmp.insert(std::make_pair(itemTmp, rankingTmp));
            rating.insert(std::make_pair(userTmp, itemRantingTmp));
        }
        else
        {
            iter->second.insert(std::make_pair(itemTmp, rankingTmp));
        }
        recordNum++;
    }
    filestream.close();
}

void initUVMatrix(Matrix& M, float mean, float variance)
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(mean, variance);

    map<string, vector<float>>::iterator iter;
    for(iter = M.matrix.begin(); iter != M.matrix.end(); iter++)
    {
        for(int j=0; j<M.column; j++)
        {
            iter->second[j] = distribution(generator);
        }
    }

}


void sgd(int train_num, int maxRating, Matrix& U, Matrix& V, int dim, float momentum, float u_lambda, float v_lambda, string filename)
{
    vector<float> loss(train_num);
    int i=0;
    while(i<train_num)
    {
        // read a record and update U/V
        std::ifstream filestream(filename.c_str());
        std::string line;
        map<string, vector<float>> userTmp;
        map<string, vector<float>> itemTmp;
        while (std::getline(filestream, line))
        {
            vector<std::string> record;
            char *p;
            char *s =(char*)line.data();
            const char *delim = "\t";
            p = strtok(s, delim);
            string user = p;
            p = strtok(NULL, delim);
            string item = p;
            p = strtok(NULL, delim);
            //float rankingTmp = std::stod(p);
            istringstream istr(p);
            float ranking = 0.0;
            float loss = 0.0;
            float predict = 0.0;
            istr>>ranking;
            // normalize to [0,1] to improve stability
            ranking = (ranking - 1) / (maxRating - 1);
            // predict R(u,v)
            map<string, vector<float>>::iterator userIter;
            map<string, vector<float>>::iterator itemIter;
            map<string, vector<float>>::iterator userIter2;
            map<string, vector<float>>::iterator itemIter2;
            userIter = U.matrix.find(user);
            if(userIter == U.matrix.end())
                std::cout<<"Do not Find, New User"<<endl;
            itemIter = V.matrix.find(item);
            if(itemIter == V.matrix.end())
                std::cout<<"Do not Find, New Item"<<endl;

            for(int i = 0; i<dim; i++)
            {
                predict += userIter->second[i]*itemIter->second[i];
            }
            // normalize
            loss = 1/(1 + exp(-1 * predict))-ranking;

            userIter2 = userTmp.find(user);
            if(userIter2 == userTmp.end())
            {
                vector<float> featureDim(dim);
                for(int i=0; i<dim; i++)
                {
                    featureDim[i] = loss*itemIter->second[i]+u_lambda*userIter->second[i];
                }
                userTmp.insert(std::make_pair(user, featureDim));
            }
            else
            {
                for(int i=0; i<dim; i++)
                {
                    userIter2->second[i] += loss*itemIter->second[i]+u_lambda*userIter->second[i];
                }
            }

            itemIter2 = itemTmp.find(item);
            if(itemIter2 == itemTmp.end())
            {
                vector<float> featureDim(dim);
                for(int i=0; i<dim; i++)
                {
                    featureDim[i] = loss*userIter->second[i]+v_lambda*itemIter->second[i];
                }
                itemTmp.insert(std::make_pair(user, featureDim));
            }
            else
            {
                for(int i=0; i<dim; i++)
                {
                    itemIter2->second[i] += loss*userIter->second[i]+v_lambda*itemIter->second[i];
                }
            }

        }
        filestream.close();

        map<string, vector<float>>::iterator userIter3;
        map<string, vector<float>>::iterator itemIter3;
        map<string, vector<float>>::iterator userIter4;
        map<string, vector<float>>::iterator itemIter4;
        for(userIter4 = userTmp.begin(); userIter4 != userTmp.end(); userIter4++)
        {
            userIter3 = U.matrix.find(userIter4->first);
            if(userIter3 != U.matrix.end())
            {
                // update u
                for(int i=0; i<dim; i++)
                {
                    userIter3->second[i] -= momentum*userIter4->second[i];
                }
            }
        }

        for(itemIter4 = itemTmp.begin(); itemIter4 != itemTmp.end(); itemIter4++)
        {
            itemIter3 = V.matrix.find(itemIter4->first);
            if(itemIter3 != V.matrix.end())
            {
                // update v
                for(int i=0; i<dim; i++)
                {
                    itemIter3->second[i] -= momentum*itemIter4->second[i];
                }
            }
        }
        i++;
    }
}


void predict(string testFilename, string outputFilename, Matrix U, Matrix V, int recordNum, int dim)
{
    std::ifstream filestream(testFilename.c_str());
    std::string line;
    ofstream outFile;
    outFile.open(outputFilename);
    while (std::getline(filestream, line))
    {
        vector<std::string> record;
        char *p;
        char *s =(char*)line.data();
        const char *delim = "\t";
        p = strtok(s, delim);
        string user = p;
        p = strtok(NULL, delim);
        string item = p;
        float predictRanking = 0.0;

        // predict R(u,v)
        map<string, vector<float>>::iterator userIter;
        map<string, vector<float>>::iterator itemIter;
        userIter = U.matrix.find(user);
        if(userIter == U.matrix.end())
            std::cout<<"Do not Find, New User"<<endl;
        itemIter = V.matrix.find(item);
        if(itemIter == V.matrix.end())
            std::cout<<"Do not Find, New Item"<<endl;

        for(int i = 0; i<dim; i++)
        {
            predictRanking += userIter->second[i]*itemIter->second[i];
        }
        // map to original ranking
        float result = predictRanking*(recordNum-1)+1;
        outFile<<result<<endl;
    }
    filestream.close();
    outFile.close();
}

/*
void minibatchProcess(string filename){
    //// shuffle
    std::ifstream filestream(filename.c_str());
    std::string line;
    while (std::getline(filestream, line))
    {
        vector<std::string> record;
        char *p;
        char *s =(char*)line.data();

        const char *delim = "\t";
        p = strtok(s, delim);
        user.insert(p);
        string userTmp = p;
        p = strtok(NULL, delim);
        item.insert(p);
        string itemTmp = p;
        p = strtok(NULL, delim);
        //float rankingTmp = std::stod(p);
        istringstream istr(p);
        float rankingTmp;
        istr>>rankingTmp;

        // normalize to [0,1] to improve stability
        rankingTmp = (rankingTmp - 1) / (maxRating - 1);

        map<string, map<string,float>>::iterator iter;
        map<string, float> itemRantingTmp;
        iter = rating.find(userTmp);
        // user does not exist in rating matrix
        if(iter == rating.end())
        {
            itemRantingTmp.insert(std::make_pair(itemTmp, rankingTmp));
            rating.insert(std::make_pair(userTmp, itemRantingTmp));
        }
        else
        {
            iter->second.insert(std::make_pair(itemTmp, rankingTmp));
        }
        recordNum++;
    }
    filestream.close();

    while(currentEpoch < maxEpoch)
    {
        currentEpoch++
        for(int i=0; i<num_batches; i++){
            // compute on each batch
            core();

        }
        // may have some records left

    }

}
*/

/*
void loadFile(Matrix& M, string filename){
std::ifstream filestream(filename.c_str());
    std::string line;
    while (std::getline(filestream, line))
    {
        vector<std::string> record;
        char *p;
        char *s =(char*)line.data();
        const char *delim = "\t";
        p = strtok(s, delim);
        while(p)
        {
            record.push_back(p);
            p = strtok(NULL, delim);
        }
        M.matrix.push_back(record);
    }
    filestream.close();
}
*/
