#ifndef BlockH
#define BlockH

//////////////////////////////////////////////////////////////////////
//
// Block.h: interface for the Block class.
//
//////////////////////////////////////////////////////////////////////


#include <vector>

#include "Ls.h"
#include "DArray.h"

class DArray;

// Rectangular section of elements

class Block {
	// std::vector of LS for every dimensions
	std::vector<LS> LSDim;

public:	

	Block(std::vector<LS> &v);
	Block();
	Block(DArray *da, long ProcLI);
	//grig
	Block(DArray *da, long ProcLI,int a); // a - �� �������� ��������
	//\grig


	virtual ~Block();

	// ��������� ��� ����������� ������� �����, ���������� ��� 
	//������������ �������� �� ���� ���������� ����� ���������� 
	long GetBlockSizeMult2(long dim1, long dim2);
	long GetBlockSizeMult(long dim);

	// true ���� ��� ������� ����� �� ������� ��������� ������ ������� 
	// ��������� ����� (������) ��������� �����. ��� ������� ����� false
	bool IsLeft(long arrDim, long elem); 
	bool IsRight(long arrDim, long elem);

	// �������� �������������� ������� ���������� �����
	bool IsBoundIn(const std::vector<long>& ALeftBSizeArray, 
		const std::vector<long>& ARightBSizeArray);

	Block & operator =(const Block & x);
	long GetRank();
	bool empty();
	 
	// ����� ��������� � �����
	long GetBlockSize();

	friend Block operator^ (Block &x, Block &y); // intersection

	//====
	long GetUpper(long i);
	long GetLower(long i);
	//=***

};

#endif 
