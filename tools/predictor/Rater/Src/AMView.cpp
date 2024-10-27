#include <assert.h>

#include "AMView.h"
#include <iostream>

using namespace std;

extern ofstream prot; 
//grig
extern long_vect MinSizesOfAM; // ��� ��������������� ������
bool FirstTrace=true;
//\grig

AMView::AMView(const vector<long>& ASizeArray) :
	Space(ASizeArray),
	Repl(0)
{
	VM_Dis		= 0;
	AlignArrays = list<DArray*>(0);
	DistRule	= vector<DistAxis>(0);
	FillArr		= vector<long>(0);
	
}

AMView::~AMView()
{
	
}

#ifdef nodef

long AMView::GetMapDim(long arrDim, int & dir)
{
	long		vmDim = 0;
	AlignAxis	align;
	DistAxis	dist;
	long		amDim;

	align = AlignRule[arrDim-1];

	if (align.Attr == align_NORMAL) {  
		amDim = align.TAxis;
		dir = (align.A > 0) ? 1 : -1;
		dist = AM_Dis->DistRule[amDim-1];
		if  (dist.Attr == map_BLOCK)
			vmDim = dist.PAxis;
	}
	return vmDim;
}

#endif

void AMView::DisAM(VM *AVM_Dis, const vector<long>& AAxisArray, const vector<long>& ADistrParamArray)
{
	int		i, 
			DistRuleSize;
	long	BlockSize;

	assert(AVM_Dis !=NULL);
	unsigned int VMR = AVM_Dis->Rank();

	assert(AAxisArray.size() <= VMR);

	DistRuleSize = VMR + Rank(); 

	FillArr = vector<long>((long) VMR, 0);

	vector<long> AxisArr;
	AxisArr.reserve(VMR);
	AxisArr = AAxisArray;
	for (i = AAxisArray.size(); i < VMR; i++)
		AxisArr[i] = 0;
	
	DistRule.reserve(DistRuleSize);

	// ��������������� �������������
	// ������ AMView.Rank() ��������� DistRule 
	for (i = 0; i < Rank(); i++)
		DistRule.push_back(DistAxis(map_COLLAPSE, i + 1, 0)); // 

	// ��������� RankOfVM  ��������� ������� DistRule 
	for (i = Rank(); i < DistRuleSize; i++)
		DistRule.push_back(DistAxis(map_NORMVMAXIS, 0, i-Rank()+1));

	//====
	for (i = 0; i < AAxisArray.size(); i++)
	{	// making a correction of array MinSizesOfAM  - for automatic finding configuration	
		if(FirstTrace==true)
		{
			//printf("Axis[%d]=%d\n",i,GetSize(AxisArr[i]));
			MinSizesOfAM[i]=MinSizesOfAM[i] > GetSize(AxisArr[i]) ? 
				MinSizesOfAM[i] : 
				GetSize(AxisArr[i]);
		}
	}
	//=***

	// ���������� DistRule � ������������ � �����������
	for (i = 0; i < VMR; i++)
	{
		if(AxisArr[i] < 0)
		{
			prot << "Wrong call DissAM" << endl;
			exit(1);
		}
		if (AxisArr[i] == 0)
			DistRule[Rank() + i] = DistAxis(map_REPLICATE, 0, i+1);
		else
		{
			DistRule[Rank() + i] = DistAxis(map_NORMVMAXIS, AxisArr[i], i+1);
			DistRule[AxisArr[i] - 1] = DistAxis(map_BLOCK, AxisArr[i], i+1);
			// ��������� � ������� � ��������� ������� 
			// � RDisAm � � RAlnDA �������� �������� ����� ������� ��� ������ ����� Repl = 1
			BlockSize = (GetSize(AxisArr[i]) - 1) / AVM_Dis->GetSize(i+1) + 1;

//printf("Blocksize=%d Bsize=%d\n",BlockSize, BSize[i]);
			FillArr[i] = AVM_Dis->GetSize(i+1) - (GetSize(AxisArr[i]) - 1) / BlockSize - 1;


//grig  making a correction of array minsizeofAm  - for automatic finding configuration
			
/* //==// moved up with some changes
			if(FirstTrace==true)
			{
				MinSizesOfAM[i]=MinSizesOfAM[i] > GetSize(AxisArr[i]) ? 
					MinSizesOfAM[i] : 
					GetSize(AxisArr[i]);
			}
*/
			//j=MinSizesOfAM[i];

			
//\grig
		}
	}
	VM_Dis = AVM_Dis;

//	vector<DistAxis>::iterator first = DistRule.begin(), last = DistRule.end();
//	cout << "i "<< "Attr "<< "Axis "<<"PAxis " <<"\n";
//	while(first != last)
//	{
//		cout << endl << first - DistRule.begin() <<  " " << first->Attr << " " << first->Axis << " " << first->PAxis << endl;
//		++first;
//	}
	#ifdef _TIME_TRACE_
	// �� �����
	vector<DistAxis>::iterator first = DistRule.begin(), last = DistRule.end();
	while(first != last)
	{
		prot << endl << first->Attr << " " << first->Axis << " " << first->PAxis << endl;
		++first;
	}
	#endif
}

double AMView::RDisAM(const vector<long>& AAxisArray, const vector<long>& ADistrParamArray, long ANewSign)
{
	if (!IsDistribute()) {
		prot << "Wrong call RDisAM" << endl;
		exit(1);
	}

	if ( ANewSign != 0) {
		// ������ ����� ��������� => �� ���� �� ����������
		DisAM(VM_Dis, AAxisArray, ADistrParamArray);
		return 0;
	}

	double time;

	DArray oldDA;
	// ��������� ���������� � ������������� AMView �� �����������������
	AMView *oldAM = new AMView(*this);
	
	// � ������� ����� ������� �� VM_Dis*VM_Dis, � VM_Dis*AVM_Dis (����� ����� ����� ������ VM)
	CommCost *rdisCost = new CommCost(VM_Dis);	
	
	// � ������� ������ VM_Dis �������� ��� ������ �������(�.�. ��������� VM ����� ����)
	DisAM(VM_Dis, AAxisArray, ADistrParamArray);

	list<DArray *>::iterator newDAi = AlignArrays.begin();
	list<DArray *>::iterator last = AlignArrays.end();

	while (newDAi != last)
	{
		oldDA = DArray(**newDAi);
		oldDA.AM_Dis = oldAM;
		rdisCost->Update(&oldDA, *newDAi);
		newDAi++;
	}

	#ifdef _TIME_TRACE_
	// ����� ������
	int i, j;
	prot << endl;
	for (i = 0; i < VM_Dis->GetLSize(); i++)
	{
		for (j = 0; j < VM_Dis->GetLSize(); j++)
		{
			prot << "[" << i << "]" << "[" << j << "] = " << rdisCost->transfer[i][j] << "; ";
		}
		prot << endl;
	}
	// ����� ������
	#endif

	time = rdisCost->GetCost();
	delete rdisCost;
	delete oldAM;
	return time; 
}


void AMView::AddDA(DArray * Aln_da)
{
	AlignArrays.push_back(Aln_da);
}

int AMView::DelDA(DArray * RAln_da)
{
	list<DArray*>::iterator i;
	i = find(AlignArrays.begin(), AlignArrays.end(), RAln_da);
	if (i == AlignArrays.end())
		return -1;
	AlignArrays.erase(i);
	return 0;
}

AMView::AMView(const AMView &x) : Space(x)
{
	DistRule = x.DistRule;
	VM_Dis = x.VM_Dis;
	AlignArrays = list<DArray*>(0);
	FillArr = x.FillArr;
}

bool AMView::IsDistribute()
{
	if (VM_Dis == 0)
		return false;
	return true;
}
