//////////////////////////////////////////////////////////////////////
//
// RedGroup.cpp: implementation of the RedGroup class.
//
//////////////////////////////////////////////////////////////////////

#include "RedGroup.h"

using namespace std;

extern ofstream prot; 
 
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RedGroup::RedGroup(VM *AvmPtr):
vmPtr(AvmPtr)
{
	redVars = vector<RedVar *>(0);
	CentralProc = vmPtr->GetCenterLI();
	TotalSize = 0;
}

RedGroup::~RedGroup()
{

}

//////////////////////////////////////////////////////////////////////
// Add reduction variable to reduction groupe
//////////////////////////////////////////////////////////////////////

void RedGroup::AddRV(RedVar * ARedVar)
{
	redVars.push_back(ARedVar);
	TotalSize += ARedVar->GetSize();
}

//////////////////////////////////////////////////////////////////////
// Calculate reduction time
//////////////////////////////////////////////////////////////////////

double RedGroup::StartR(DArray *APattern, long ALoopRank, const vector<long>& AAxisArray)
{
	double	time = 0;
	long	i, 
			redBlSize = 1, 
			redBlCenterDist = 0;
	vector<long> loopAlign(ALoopRank);
	int		dir;
	long	LSize = vmPtr->GetLSize();
	bool	redBlEmpty = true;

	// ???��� ������-�� ������, �.�. ���.���������� ���������� �� ���� ����. ������, 
	// � ������ ������ �� ����. ������, � ��� �� ������ ���������
	// � ��� �� ��� ������ ���������� � ����� ���� ���������, �� ��� 
	// � ������ ������� Update
	// ��� �������� �� ����� ��� ��� �� ����� �� ��� ���������� ���� ��������
	// (���� ������� �� �� ���� ����.������ ��������� ����)

	/*
	if (APattern->Repl)
		return 0; 

	for (i = 0; i < ALoopRank; i++) {
		loopAlign[i] = APattern->GetMapDim(AAxisArray[i], dir);
//		prot << "loopAlign[" << i << "] = " << loopAlign[i] << endl;
	}
	*/


//	printf("WANNA RED %d\n",TotalSize);

	// ����������� - ������� (� ���� ���� ����������)
	for(i=1;i<MPSProcCount();i++)
	{
		Block locBlock(APattern, i,1);

		if(locBlock.empty()) 
			continue;
			
		CommCost cc;
		cc.CommSend(i,0,TotalSize);

//		printf("Curr_pt=%f  Beg_time=%f   End_time=%f\n",CurrProcTime(currentVM->map(i)), cc.BeginTime, cc.EndTime);

	}

	for(i=1;i<MPSProcCount();i++)
	{
		Block locBlock(APattern, i,1);

		if(locBlock.empty()) 
			continue;
			
		CommCost cc;
		cc.CommSend(0,i,TotalSize);
//		printf("!!Curr_pt=%f  Beg_time=%f   End_time=%f\n",CurrProcTime(currentVM->map(i)), cc.BeginTime, cc.EndTime);

		if(time < cc.EndTime) 
			time=cc.EndTime;
	}

	return time; // ���������� ����� ����� ��������, � ������ ���� ������ ������ �����
}

//////////////////////////////////////////////////////////////////////
// Calculate reduction time
//////////////////////////////////////////////////////////////////////

// ��� ����� �������� �� AMView � ��� ��� ��������� �� ������
double RedGroup::StartR(AMView *APattern, long ALoopRank, const vector<long>& AAxisArray)
{
	double time = 0;
	long i, redBlSize = 1, redBlCenterDist = 0;
	vector<long> loopAlign(ALoopRank);
//	int dir;
	long LSize = vmPtr->GetLSize();
//	double TStart = vmPtr->getTStart();
//	double TByte = vmPtr->getTByte();
	bool redBlEmpty = true;

	// ???��� ������-�� ������, �.�. ���.���������� ���������� �� ���� ����. ������,
	// � ������ ������ �� ����. ������, � ��� �� ������ ���������
	// � ��� �� ��� ������ ���������� � ����� ���� ���������, �� ��� � 
	// ������ ������� Update
	// ��� �������� �� ����� ��� ��� �� ����� �� ��� ���������� ���� ��������
	// (���� ������� �� �� ���� ����.������ ��������� ����)



	return time;
}

