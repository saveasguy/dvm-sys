#ifndef DimBoundH
#define DimBoundH

//////////////////////////////////////////////////////////////////////
//
// DimBound.h: interface for the DimBound class.
//
//////////////////////////////////////////////////////////////////////



class DimBound {
public:
	long arrDim;		// Array dimension
	long vmDim;			// Virtual machine dimension
	int dir;			// ����� 1 ��� -1 � ����������� �� ����������� 
						// ��������� ��������� �������
	long LeftBSize;
	long RightBSize;

	DimBound(long AarrDim, long AvmDim, int Adir, long ALeftBSize, long ARightBSize);
	DimBound();
	virtual ~DimBound();
};

bool operator==(const DimBound& x, const DimBound& y);

bool operator<(const DimBound& x, const DimBound& y);

#endif
