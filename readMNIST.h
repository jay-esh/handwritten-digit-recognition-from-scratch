#include "Eigen/Dense"

using Eigen::MatrixXd;

const int nRow = 28;
const int nColumn = 28;
struct character
{
	unsigned char Label, Image[nRow][nColumn];
};

MatrixXd getIntValMatrix(int iCharacter, character *Character);
void printCharArray(MatrixXd array);
MatrixXd getLabelMx(int iCharacter, character *Character);
void SwapEndian32(int *v);
void readInt(std::ifstream &File, int *MyInt, bool bSwapEndian);
int readMNIST(std::string ImageFileName, std::string LabelFileName,
			  int &nCharacter, character *&Character);
unsigned char LabelToText(unsigned char Label);
unsigned char PixelToText(unsigned char Pixel);
void printImage(int iCharacter, int nCharacter, character *Character);
