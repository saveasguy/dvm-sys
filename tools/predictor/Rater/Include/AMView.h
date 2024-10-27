#ifndef AMViewH
#define AMViewH

//////////////////////////////////////////////////////////////////////
//
// AMView.h: interface for the AMView class.
//
//////////////////////////////////////////////////////////////////////

#include <stdlib.h>

#include <fstream>
#include <list>
#include <vector>
#include <algorithm>

#include "Vm.h"
#include "DistAxis.h"
#include "DArray.h"

class DArray;


//grig
class WeightClass
{
public :
	long ID;				     // PS , ��� ����������� � �������  ����� �������������� ���� 
	std::vector<double> body;    // opt weights
	
	WeightClass()
	{
		ID=NULL;
		body.resize(0);
	}

	WeightClass(long AID,std::vector<double>& init_weights)
	{
		ID=AID;
		body.resize(0);
		for(int i=0;i<init_weights.size();i++)
		{
			body.push_back(init_weights[i]);
		}

	}

	void GetWeights(std::vector<double> & AAweights)
	{
//		printf("Get SZ id=%lx %d\n",ID,body.size());
		AAweights.resize(body.size());
		for(int i=0;i<body.size();i++)
			AAweights[i]=body[i];
	}

	void SetWeights(long AID,std::vector<double>& init_weights)
	{// printf("Set SZ id=%lx %d\n",ID,init_weights.size());
	
		ID=AID;
		body.resize(0);
		for(int i=0;i<init_weights.size();i++)
		{
			body.push_back(init_weights[i]);
		}
	}
	 ~WeightClass()
	 {
	   body.resize(0);	 
	 }

	 long GetSize()
	 {
		 return body.size();	 
	 }
};
//\grig
class AMView : public Space {

public:	 
	VM *VM_Dis;							// VM �� ������� ���������� AMView
	std::list<DArray*> AlignArrays;		// ������ ���������� ��������
	std::vector<DistAxis> DistRule;		// ��������� ����������� � ����������� ������
	std::vector<long> FillArr;			// ??? 
	std::vector<long> BSize;		//====// ����� ����� ������ ����� ������� 

	int Repl;							// ������� ��������� ������������� �� AM_Dis ������� 

//grig
	WeightClass weightEl;           // ���� ��� ����������� �������
//\grig
 
 	AMView(const std::vector<long>& ASizeArray);
	AMView(const AMView &);			// !!! ���������� ������ VM_Dis � DistRule � FillArr
    ~AMView();

	long	GetMapDim(long arrDim, int & dir);
	int		DelDA(DArray* RAln_da);		// ������� DArray �� ������ AlignArrays
	void	AddDA(DArray* Aln_da);		// �������� DArray � ������ AlignArrays

    // ����������� ������������� ����������� ������ � ����������� ������
 	void DisAM(VM *AVM_Dis, const std::vector<long>& AAxisArray, const std::vector<long>& ADistrParamArray);

    // ��������� ����������� ������������� ����������� ������ � ����������� ������
    double RDisAM(const std::vector<long>& AAxisArray, const std::vector<long>& ADistrParamArray, 
		long ANewSign);

	bool IsDistribute();
};





#endif
