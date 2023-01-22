#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include "readMNIST.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;

void SwapEndian32(int *v)
{
  unsigned int *value = (unsigned int *)v;
  unsigned int tmp = ((*value << 8) & 0xFF00FF00) | ((*value >> 8) & 0xFF00FF);
  *value = (tmp << 16) | (tmp >> 16);
}

void readInt(std::ifstream &File, int *MyInt, bool bSwapEndian)
{
  File.read((char *)MyInt, sizeof(int));
  // std::cout << "int in: " << *MyInt;
  if (bSwapEndian)
    SwapEndian32(MyInt);
  // std::cout << " Endian " << *MyInt << "\n";
}

int readMNIST(std::string ImageFileName, std::string LabelFileName,
              int &nCharacter, character *&Character)
{

  std::ifstream ImageFile, LabelFile;
  int nImage, nLabel, magic, nRowIn, nColumnIn;
  bool bSwapEndian = false;

  ImageFile.open(ImageFileName.c_str(), std::ios::binary);
  if (!ImageFile.is_open())
  {
    std::cout << "Could not open file: " << ImageFileName << "\n";
    exit(-1);
  }
  ImageFile.read((char *)&magic, sizeof(int));
  if (magic != 2051)
  {
    SwapEndian32(&magic);
    if (magic == 2051)
    {
      bSwapEndian = true;
    }
    else
    {
      std::cout << "This is not an Image file: " << ImageFileName << "\n";
      exit(-1);
    }
  }
  readInt(ImageFile, &nImage, bSwapEndian);
  readInt(ImageFile, &nRowIn, bSwapEndian);
  readInt(ImageFile, &nColumnIn, bSwapEndian);
  if (nRowIn != nRow || nColumnIn != nColumn)
  {
    std::cout << "Format of Images " << nRowIn << "x" << nColumnIn
              << ".  Expected " << nRow << "x" << nColumn << "\n";
    exit(-1);
  }

  LabelFile.open(LabelFileName.c_str(), std::ios::binary);
  if (!LabelFile.is_open())
  {
    std::cout << "Could not open file: " << LabelFileName << "\n";
    exit(-1);
  }
  readInt(LabelFile, &magic, bSwapEndian);
  if (magic != 2049)
  {
    std::cout << "This is not an Label file: " << LabelFileName << "\n";
    exit(-1);
  }
  readInt(LabelFile, &nLabel, bSwapEndian);

  if (nImage != nLabel)
  {
    std::cout << "nImage = " << nImage << " != nLabel = " << nLabel << "\n";
    exit(-1);
  }
  // std::cout << "nImage:" << nImage << "\n";
  nCharacter = nImage;
  // std::cout << "nImageChar:" << nCharacter << "\n";
  Character = new character[nCharacter]; // allocate data storage
  //*pCharacter = Character;  // save location of storage into original array

  for (int i = 0; i < nCharacter; i++)
  {
    if (!LabelFile.read((char *)&(Character[i].Label), sizeof(unsigned char)) ||
        !ImageFile.read((char *)&(Character[i].Image), nRow * nColumn * sizeof(unsigned char)))
    {
      std::cout << "Read incomplete for entry " << i << "\n";
      exit(-1);
    }
  }
  LabelFile.close();
  ImageFile.close();

  std::cout << "Read Complete: " << nCharacter << " Labels and " << nRow << "x" << nColumn << " Images\n";

  return nCharacter;
}

unsigned char LabelToText(unsigned char Label)
{
  return (unsigned char)('0' + Label);
}

unsigned char PixelToText(unsigned char Pixel)
{
  if (Pixel < 64)
  {
    return ' ';
  }
  if (Pixel < 128)
  {
    return '-';
  }
  if (Pixel < 192)
  {
    return '=';
  }
  return '#';
}

void printImage(int iCharacter, int nCharacter, character *Character)
{
  for (int j = 0; j < nRow; j++)
  {
    for (int i = 0; i < nColumn; i++)
    {
      std::cout << PixelToText(Character[iCharacter].Image[j][i]);
    }
    std::cout << "\n";
  }
}

MatrixXd getIntValMatrix(int iCharacter, character *Character)
{
  // int **array = new int *[nRow];
  MatrixXd array(1, (nRow * nColumn));
  int pointer = 0;
  for (int i = 0; i < nRow; i++)
  {
    // array[i] = new int[nColumn];

    for (int j = 0; j < nColumn; j++)
    {
      array(0, pointer) = (double)(+Character[iCharacter].Image[i][j]) / 255.0;
      pointer++;
    }
  }
  // std::cout << "size: " << array.cols() << std::endl;
  return array;
}

MatrixXd getLabelMx(int iCharacter, character *Character)
{
  double input = 1.0;
  MatrixXd retLabel(1, 10);
  int label = (int)(+Character[iCharacter].Label);
  for (int i = 0; i < 10; i++)
  {
    if (i == label)
    {
      retLabel(0, i) = input;
    }
    else
    {
      retLabel(0, i) = 0.0;
    }
  }
  // std::cout << retLabel << std::endl;
  return retLabel;
}

void printCharArray(MatrixXd array)
{
  std::cout << array << std::endl;
}

// void printImage(int iCharacter, int nCharacter, character *Character)
// {
//   for (int j = 0; j < nRow; j++)
//   {
//     for (int i = 0; i < nColumn; i++)
//     {
//       std::cout << PixelToText(Character[iCharacter].Image[j][i]);
//     }
//     std::cout << "\n";
//   }
// }
