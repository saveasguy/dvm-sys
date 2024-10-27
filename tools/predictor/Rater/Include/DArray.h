#ifndef DArrayH
#define DArrayH

//////////////////////////////////////////////////////////////////////
//
// DArray.h: interface for the DArray class.
//
//////////////////////////////////////////////////////////////////////


#include <vector>
#include <fstream>

#include "Space.h"
#include "AMView.h"
#include "AlignAxis.h"
#include "Block.h"
#include "CommCost.h"
//#include "ModelStructs.h"

class AMView;

class DArray : public Space {
	void PrepareAlign(long& TempRank, const std::vector<long>& AAxisArray, 
		const std::vector<long>&  ACoeffArray, const std::vector<long>& AConstArray, 
		std::vector<AlignAxis>& IniRule);

	
	//  ��������� ������� �� ����� �� ������� �������. 
	// ������������ ����� ��������� � �����. 0 ���� �� ��� ��� ����� �� ������
	long CheckIndex(const std::vector<long>& InitIndexArray, 
		std::vector<long>& LastIndexArray, 
		const std::vector<long>& StepArray);

public:

	std::vector<long>	LowShdWidthArray;
	std::vector<long>	HiShdWidthArray;
	long TypeSize;		// ������ � ������ ������ �������� �������
	AMView *AM_Dis;		// AMView  � ������� ������������ DArray
	// ������� ������������ �� AM_Dis - ? 
	// ��� ���� ����� - ��� Pattern ��� ��� ��� ����-������ ����
	std::vector<AlignAxis> AlignRule; 
	int Repl; // ������� ��������� ������������� �� AM_Dis ������� 

	DArray();
 	DArray(const std::vector<long>& ASizeArray, const std::vector<long>& ALowShdWidthArray,
		const std::vector<long>& AHiShdWidthArray, int ATypeSize);
	DArray(const DArray &);
    ~DArray();
	DArray & operator= (const DArray &x);

	void AlnDA(AMView *APattern, const std::vector<long>& AAxisArray, 
		const std::vector<long>& ACoeffArray, const std::vector<long>& AConstArray);
    void AlnDA(DArray* APattern,  const std::vector<long>& AAxisArray, 
		const std::vector<long>& ACoeffArray, const std::vector<long>& AConstArray);

    double RAlnDA(AMView *APattern,  const std::vector<long>& AAxisArray, 
		const std::vector<long>& ACoeffArray, const std::vector<long>& AConstArray, 
		long ANewSign);
    double RAlnDA(DArray* APattern,  const std::vector<long>& AAxisArray, 
		const std::vector<long>& ACoeffArray, const std::vector<long>& AConstArray, 
		long ANewSign);

    friend double ArrayCopy(
		DArray* AFromArray, 
		const std::vector<long>& AFromInitIndexArray, 
		const std::vector<long>& AFromLastIndexArray, 
		const std::vector<long>& AFromStepArray, 
		DArray* AToArray, 
		const std::vector<long>& AToInitIndexArray, 
		const std::vector<long>& AToLastIndexArray, 
		const std::vector<long>& AToStepArray); 

    friend double ArrayCopy(DArray* AFromArray, 
		const std::vector<long>& AFromInitIndexArray, 
		const std::vector<long>& AFromLastIndexArray, 
		const std::vector<long>& AFromStepArray, 
		long ACopyRegim);


    // ArrCpy ? - ����� ���������, ��� ������

	long GetMapDim(long arrDim, int &dir); // ������� ���������� ����� 
	// ��������� VM �� ������� ���������� ��������� ��������� �������
	// (���� ��������� ������� ���������� �� ���� ������������ ������� ����������� ����������� - 0).
	// � dir ��������� 1 ��� -1 � ����������� �� ����������� ��������� ��������� �������
	bool IsAlign();

	double RDisDA(const std::vector<long>& AAxisArray, const std::vector<long>& ADistrParamArray, 
					long ANewSign);
#ifdef P_DEBUG
	friend std::ostream& operator << (std::ostream& os, const DArray& s);
#endif

};


#endif
