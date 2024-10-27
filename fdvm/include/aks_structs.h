#pragma once
#include "acc_data.h"

struct SageSymbols
{
    SageSymbols()
    {
        across_left = across_right = 0;
        len = -1;
        symb = NULL;
    }

    SageSymbols(SgSymbol* symb, int len, int across_left, int across_right) :
        symb(symb), len(len), across_left(across_left), across_right(across_right)
    { }

    SgSymbol *symb;
    int len;
    int across_left;
    int across_right;
};

struct SageArrayIdxs
{
    std::vector<SageSymbols> symb;
    int dim;
    int read_write;
    SgExpression *array_expr;
};

struct SageAcrossInfo
{
    std::vector<SageArrayIdxs> idxs;
};

struct ArgsForKernel
{
    SgStatement *st_header;
    std::vector<SageSymbols> symb;
    std::vector<SageSymbols> nSymb;
    std::vector<SgSymbol*> sizeVars;
    std::vector<SgSymbol*> acrossS;
    std::vector<SgSymbol*> notAcrossS;
    std::vector<SgSymbol*> idxAcross;
    std::vector<SgSymbol*> idxNotAcross;

    std::vector<SgSymbol*> otherVars;
    std::vector<char*> arrayNames;
    std::vector<SgSymbol*> otherVarsForOneTh;
    std::vector<SgSymbol*> baseIdxsInKer;
    SgSymbol *cond_;
    std::vector<SgSymbol*> steps;
};

/*struct GetXYInfo
{
    std::vector<SgExpression*> AllExp;
    SgSymbol *varName;
    char *arrayName;
    long type;
    int placeF;
    int placeS;
};*/


/*struct TransformInfo
{
std::vector<char *> arrayNames;
std::vector<int> firstIdxs;
std::vector<int> secondIdxs;
bool exist(char *name);
bool find_in_exprs(SgSymbol*, SgSymbol*, char*, SgExpression*);
bool find_add(SgSymbol*, SgSymbol*, char*);
void getIdxs(char*, int&, int&);
};*/

struct ParamsForAllVariants
{
    SgSymbol *s_adapter;
    SgSymbol *s_kernel_symb;
    int loopV;
    int acrossV;
    int allDims;
    std::vector<SageSymbols> loopSymb;
    std::vector<SageSymbols> loopAcrossSymb;
    char *nameOfNewSAdapter;
    char *nameOfNewKernelSymb;
    int type;
};

struct Bound
{
    int L;
    int R;
    bool exL;
    bool exR;
    bool ifDdot;
    SgExpression *additionalExpr;
};

struct BestPattern
{
    std::vector<int> what;
    std::vector<Bound> bounds;
    SgExpression *bestPatt;
    int count_of_pattern;
};

struct Pattern
{
    int count_read_op;
    int count_write_op;
    SgExpression *symbs;
};

struct AnalyzeStat
{
    SgSymbol *replaceSymbol;
    int ifHasDim;
    SgSymbol *name_of_array;
    SgExpression *ex_name_of_array;
    std::vector<Pattern> patterns;
};


// <for oprimization>
struct acrossInfo
{
    char *nameOfArray;
    SgSymbol *symbol;
    int allDim;
    int acrossPos;
    int widthL;
    int widthR;
    int acrossNum;
    std::vector<int> dims;
    std::vector<SgSymbol*> symbs;
};

struct newInfo
{
    SgSymbol *newArray;
    std::vector<int> dimSize;
    std::vector<SgStatement*> loadsBeforePlus;
    std::vector<SgStatement*> loadsInForPlus;
    std::vector<SgStatement*> loadsBeforeMinus;
    std::vector<SgStatement*> loadsInForMinus;
    std::vector<SgStatement*> stores;
    std::vector<SgStatement*> swapsDown;
    std::vector<SgStatement*> swapsUp;
};
// end <for oprimization>

// block <gpuO1 lvl 2>
struct Group
{
    char *strOfmain; //
    SgExpression *mainPattern;
    std::vector<SgExpression*> inGroup;
    std::vector<int> len;
    std::vector<int> sortLen;
    newInfo replaceInfo; // replace info with all needed loads and swaps for optimization
};

struct PositionGroup
{
    std::map<std::string, SgExpression*> tableReplace; // table of mapping new private variables to distributed arrays for replacing in loop body
    std::map<std::string, SgSymbol*> tableNewVars; // table of new private variables that is needed to add in cuda kernel
    int position; // position of fixed variable in distributed loop, index 0 corresponds to the first variable.
    SgExpression *idxInPos; //
    std::vector<Group> allPosGr; // all groups of array access patterns with fixed loop variables, which is distributed
};

struct ArrayGroup
{
    SgSymbol *arrayName; // name of distribute array
    std::vector<PositionGroup> allGroups; // all groups, where one loop variable is fixed
};
// end of block <gpuO1 lvl 2>

struct LoopInfo
{
    std::vector<SgSymbol*> loopSymbols;
    std::vector<SgExpression*> lowBounds;
    std::vector<SgExpression*> highBounds;
    std::vector<SgExpression*> steps;
    int lineNumber;
};

struct ArrayIntents
{
    std::vector<SgSymbol*> arrayList;
    std::vector<int> intent;
};

struct AnalyzeReturnGpuO1
{
    std::vector<AnalyzeStat> allStat;
    std::vector<BestPattern> bestPatterns;
    std::vector<ArrayGroup> allArrayGroup;
};

// functions
SgExpression* findDirect(SgExpression*, int);
SageAcrossInfo GetLoopsWithParAndAcrDir();
std::vector<SageSymbols> GetSymbInParalell(SgExpression*);
int GetIdxPlaceInParDir(SageSymbols*, SgSymbol*);
