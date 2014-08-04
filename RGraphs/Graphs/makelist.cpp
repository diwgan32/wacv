#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main(int argc, char*argv[]){
	ifstream fin;
	fin.open("verif.txt");
	int count = 0;

	ofstream fout;
	fout.open("output.txt");
	for(int i = 0; i<2; i++){
		for(int j = 0; j<2; j++){
			for(int k = 0; k<3; k++){
				for(int l = 0; l<3; l++){
					if(count < 24){
						string str = "";
						str = (i==0 ? "-" : "+");
						str+=	(j==0 ? "-" : "+");
						str+=	(k==0 ? "-" : k==1 ? "0" : "+");
						str+=	(l==0 ? "-" : l==1 ? "0" : "+");
						double verf;
						fin >> verf;
						fout << str << "\t" << verf << endl;
					}
					count++;
				}
			}
		}
	}
	fin.close();
	fout.close();
	return 0;
}