/*

Test of output-to-input neural net on some simple operations.

*/

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <ctime>

/*
//network details

#define numlayers 2
#define maxNodes 40
#define startingParameterRange 0.1

//training deatils

#define learnRate 0.001
#define momentum 0
#define batchSize 10
#define numEval 5000

using namespace std;

ofstream netOut("addmult.txt");
ifstream netIn("addmult.txt");
ofstream dataOut("addmult_data.txt");

const int seqLength = 3;
const int numInputs = 1;
const int numOutputs = 1;

const int numCarry = 1;

int numNodes[numlayers+1] = {numInputs + numCarry, numOutputs + numCarry};

void generateData(double* inputs, double* outputs){
    int sum = 0;
    for(int i=0; i<seqLength; i++){
        inputs[i] = rand() % 10;
        sum += inputs[i];
        outputs[i] = sum;
    }
}
*/

//network details

#define numlayers 2
#define maxNodes 40
#define startingParameterRange 0.1

//training deatils

#define learnRate 0.001
#define momentum 0
#define batchSize 4
#define numEval 100
#define numTrain 300000

using namespace std;

ofstream netOut("addmult.txt");
ifstream netIn("saved_net.txt");
ofstream dataOut("addmult_data.txt");

const int seqLength = 100;
const int numInputs = 5;
const int numOutputs = 5;

const int numCarry = 6;

int numNodes[numlayers+1] = {numInputs + numCarry, 15, numOutputs + numCarry};

void generateData(double* inputs, double* outputs){
    bool bits[numInputs]{};
    for(int i=0; i<seqLength; i++){
        for(int j=0; j<numInputs; j++){
            int bit = rand() % 2;
            inputs[i*numInputs + j] = bit;
            bits[j] ^= bit;
            outputs[i*numInputs + j] = bits[j];
        }
    }
}

class Network{
public:
    double weights[numlayers][maxNodes][maxNodes];
    double bias[numlayers][maxNodes];
    double activation[numlayers+1][maxNodes];
    double inter[numlayers][maxNodes];
    double output[numOutputs];
    
    double expectedOutput[numOutputs];
    double Dbias[numlayers][maxNodes];
    double Dactivation[numlayers+1][maxNodes];
    
    void randomize(){
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l]; i++){
                for(j=0; j<numNodes[l+1]; j++){
                    weights[l][i][j] = randVal();
                }
            }
        }
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                bias[l][i] = randVal();
            }
        }
    }
    
    void nudge(double magnitude){
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l]; i++){
                for(j=0; j<numNodes[l+1]; j++){
                    weights[l][i][j] += randVal() * randVal() * magnitude;
                }
            }
        }
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                bias[l][i] += randVal() * randVal() * magnitude;
            }
        }
    }
    
    void copy(Network* net){
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l]; i++){
                for(j=0; j<numNodes[l+1]; j++){
                    weights[l][i][j] = net->weights[l][i][j];
                }
            }
        }
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                bias[l][i] = net->bias[l][i];
            }
        }
    }
    
    void pass(){
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                inter[l][i] = bias[l][i];
                for(j=0; j<numNodes[l]; j++){
                    inter[l][i] += weights[l][j][i] * activation[l][j];
                }
                activation[l+1][i] = nonlinear(inter[l][i], l);
            }
        }
        for(i=0; i<numNodes[numlayers]; i++){
            output[i] = activation[numlayers][i];
        }
    }
    
    void backProp(){
        int l,i,j;
        for(l=numlayers-1; l>=0; l--){
            for(i=0; i<numNodes[l+1]; i++){
                Dbias[l][i] = Dactivation[l+1][i] * dnonlinear(inter[l][i], l);
//                cout << "dbias " << Dactivation[l+1][i] * dnonlinear(inter[l][i], l) << endl;
            }
            for(i=0; i<numNodes[l]; i++){
                Dactivation[l][i] = 0;
                for(j=0; j<numNodes[l+1]; j++){
                    Dactivation[l][i] += Dbias[l][j] * weights[l][i][j];
//                    cout << "Dactivation " << Dbias[l][j] * weights[l][i][j] << endl;
                }
            }
        }
    }
    
    void saveNet(){
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l]; i++){
                for(j=0; j<numNodes[l+1]; j++){
                    netOut<<weights[l][i][j]<<' ';
                }
                netOut<<'\n';
            }
            netOut<<'\n';
        }
        netOut<<'\n';
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                netOut<<bias[l][i]<<' ';
            }
            netOut<<'\n';
        }
    }
    
    void readNet(){
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l]; i++){
                for(j=0; j<numNodes[l+1]; j++){
                    netIn>>weights[l][i][j];
                }
            }
        }
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                netIn>>bias[l][i];
            }
        }
    }
    
private:
    double randVal(){
        return (((double)rand() / RAND_MAX)*2-1) * startingParameterRange;
    }
    /*
    double nonlinear(double x){
        return 1/(1+exp(-x));
    }
    
    double dnonlinear(double x){
        return nonlinear(x) * (1-nonlinear(x));
    }
    */
    
    double nonlinear(double x, int l){
        if(x>0) return x;
        return 0.1 * x;
    }
    
    double dnonlinear(double x, int l){
        if(x>0) return 1;
        return 0.1;
    }
};

class NetworkSeq{
public:
    Network params;
    
    Network nets[seqLength];
    
    double Sbias[numlayers][maxNodes];
    double Sweights[numlayers][maxNodes][maxNodes];
    
    void randomize(){
        params.randomize();
    }
    
    void pass(double* inputs){
        for(int i=0; i<seqLength; i++){
            nets[i].copy(&params);
            //nets[i] = params;
        }
        for(int n=0; n<seqLength; n++){
            memcpy(nets[n].activation[0], inputs + n*numInputs, numInputs * sizeof(double));
            if(n == 0){
                for(int i=0; i<numCarry; i++){
                    nets[0].activation[0][numInputs + i] = 0;
                }
            }
            else{
                memcpy(nets[n].activation[0] + numInputs, nets[n-1].activation[numlayers] + numOutputs, numCarry * sizeof(double));
            }
            nets[n].pass();
        }
    }
    
    void backProp(double* inputs, double* expected){
        pass(inputs);
        for(int n=seqLength-1; n>=0; n--){
            for(int i=0; i<numOutputs; i++){
                nets[n].Dactivation[numlayers][i] = 2 * (nets[n].activation[numlayers][i] - expected[n*numOutputs + i]);
            }
            if(n == seqLength-1){
                for(int i=0; i<numCarry; i++){
                    nets[n].Dactivation[numlayers][numOutputs + i] = 0;
                }
            }
            else{
                memcpy(nets[n].Dactivation[numlayers] + numOutputs, nets[n+1].Dactivation[0] + numInputs, numCarry * sizeof(double));
            }
            nets[n].backProp();
        }
        for(int n=0; n<seqLength; n++){
            int l,i,j;
            for(l=0; l<numlayers; l++){
                for(i=0; i<numNodes[l]; i++){
                    for(j=0; j<numNodes[l+1]; j++){
                        Sweights[l][i][j] += nets[n].Dbias[l][j] * nets[n].activation[l][i];
                    }
                }
            }
            for(l=0; l<numlayers; l++){
                for(i=0; i<numNodes[l+1]; i++){
                    Sbias[l][i] += nets[n].Dbias[l][i];
                }
            }
        }
    }
    
    void initSum(){
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l]; i++){
                for(j=0; j<numNodes[l+1]; j++){
                    Sweights[l][i][j] = 0;
                }
            }
        }
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                Sbias[l][i] = 0;
            }
        }
    }
    
    void update(double error){
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l]; i++){
                for(j=0; j<numNodes[l+1]; j++){
                    params.weights[l][i][j] -= Sweights[l][i][j] * learnRate / batchSize;
                    Sweights[l][i][j] *= momentum;
                    
                }
            }
        }
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                params.bias[l][i] -= Sbias[l][i] * learnRate / batchSize;
                Sbias[l][i] *= momentum;
            }
        }
    }
};

NetworkSeq net;
double testInput[numInputs*seqLength];
double testOutput[numOutputs*seqLength];

double squ(double x){
    return x*x;
}

double evaluate(){
    double error = 0;
    for(int i=0; i<numEval; i++){
        generateData(testInput, testOutput);
        net.pass(testInput);
        for(int j=0; j<seqLength; j++){
            for(int k=0; k<numOutputs; k++){
                error += abs(net.nets[j].activation[numlayers][k] - testOutput[j*numOutputs + k]);
            }
        }
    }
    cout<<"Average Error: "<<(error / numEval / seqLength / numOutputs)<<'\n';
    return error / numEval / seqLength / numOutputs;
}

double getError(){
    double error = 0;
    for(int j=0; j<seqLength; j++){
        for(int k=0; k<numOutputs; k++){
            error += squ(net.nets[j].activation[numlayers][k] - testOutput[j*numOutputs + k]);
        }
    }
    return error;
}

int main(){
    srand((unsigned int) time(0));
    
    /*
    net.params.readNet();
    
    generateData(testInput, testOutput);
    net.pass(testInput);
    for(int j=0; j<seqLength; j++){
        cout << "AI says: ";
        for(int k=0; k<numOutputs; k++){
            cout << net.nets[j].activation[numlayers][k] << " ";
        }
        cout << endl;
        cout << "Reality says: ";
        for(int k=0; k<numOutputs; k++){
            cout << testOutput[j*numOutputs + k] << " ";
            
        }
        cout << endl << endl;
    }
    */
    
    
    net.randomize();
    net.initSum();
    
    double error = 1;
    
    for(int i=0; i<numTrain; i++){
        for(int j=0; j<batchSize; j++){
            generateData(testInput, testOutput);
            net.backProp(testInput, testOutput);
        }
        net.update(error);
        if(i%10000 == 0){
            error = evaluate();
        }
    }
    
    net.params.saveNet();/**/
     
    
//    testInput[0] = 3;
//    testInput[1] = 1;
//    testInput[2] = 5;
//    testOutput[0] = 4;
//    testOutput[1] = 3;
//    testOutput[2] = 7;
//
//    net.params.weights[0][0][0] = 2;
//    net.params.weights[0][0][1] = 1;
//    net.params.weights[0][1][0] = 1;
//    net.params.weights[0][1][1] = -1;
//    net.params.bias[0][0] = 3;
//    net.params.bias[0][1] = 2;
    
    /*
    net.pass(testInput);
    
    for(int i=0; i<seqLength; i++){
        for(int j=0; j<numOutputs; j++){
            cout<<net.nets[i].activation[numlayers][j]<<' ';
        }
        cout<<'\n';
    }
     */
//    net.backProp(testInput, testOutput);
    
    /*
    int l,i,j;
    for(l=0; l<numlayers; l++){
        for(i=0; i<numNodes[l]; i++){
            for(j=0; j<numNodes[l+1]; j++){
                net.backProp(testInput, testOutput);
                double netDer =
                double initError = getError();
                net.params.weights[l][i][j] += 0.01;
                net.backProp(testInput, testOutput);
                double newError = getError();
                net.params.weights[l][i][j] -= 0.01;
                double derivative = (newError - initError) / 0.01;
            }
        }
    }
     */
}
