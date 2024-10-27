/*********************************************************************/
/*  Fortran DVM+OpenMP+ACC                                           */
/*                                                                   */
/*                   Call Site Processing                            */
/*********************************************************************/
#include "leak_detector.h"

#include "dvm.h"
#include "acc_data.h"
#include "calls.h"

using std::map;
using std::string;
using std::vector;
using std::pair;

//---------------------------------------------------------------------------------

#define NEW  1
#define STATIC 1

graph_node *cur_node;
graph_node *node_list;
int deb_reg = 0;
int do_dummy = 0;
int do_stmtfn = 0;
int gcount = 0;
int has_generic_interface = 0;
int in_region = 0;
//-----------------------------------------------------------------------------------------
graph_node *GraphNode(SgSymbol *s, SgStatement *header_st, int flag_new);
graph_node *NodeForSymbInGraph(SgSymbol *s, SgStatement *stheader);
graph_node *NewGraphNode(SgSymbol *s, SgStatement *header_st);
edge *CreateOutcomingEdge(graph_node *gnode, int inlined);
edge *CreateIncomingEdge(graph_node *gnode, int inlined);
edge *NewEdge(graph_node *from, graph_node *to, int inlined);
int isDummyArgument(SgSymbol *s);
int isHeaderStmtSymbol(SgSymbol *s);
int isStatementFunction(SgSymbol *s);
int isHeaderNode(graph_node *gnode);
int isDeadNode(graph_node *gnode);
int isNoBodyNode(graph_node *gnode);
void PrototypeOfFunctionFromOtherFile(graph_node *node, SgStatement *after);
graph_node_list  *addToNodeList(graph_node_list *pnode, graph_node *gnode);
graph_node_list  *delFromNodeList(graph_node_list *pnode, graph_node *gnode);
graph_node_list  *isInNodeList(graph_node_list *pnode, graph_node *gnode);
void PrintGraphNode(graph_node *gnode);
void PrintGraphNodeWithAllEdges(graph_node *gnode);
void PrintWholeGraph();
void PrintWholeGraph_kind_2();
void BuildingHeaderNodeList();
void RemovingDeadSubprograms();
void NoBodySubprograms();
void DeleteIncomingEdgeFrom(graph_node *gnode, graph_node *from);
void DeleteOutcomingEdgeTo(graph_node *gnode, graph_node *gto);
void ScanSymbolTable(SgFile *f);
void ScanTypeTable(SgFile *f);
void printSymb(SgSymbol *s);
void printType(SgType *t);
//-------------------------------------------------------------------------------------
extern SgExpression *private_list;
extern map <string, vector<vector<SgType*> > > interfaceProcedures;

void MarkAsUserProcedure(SgSymbol *s)
{
    SYMB_ATTR(s->thesymb) = SYMB_ATTR(s->thesymb) | USER_PROCEDURE_BIT;
}

void MarkAsExternalProcedure(SgSymbol *s)
{
    SYMB_ATTR(s->thesymb) = SYMB_ATTR(s->thesymb) | EXTERNAL_BIT;
}

SgSymbol * GetProcedureHeaderSymbol(SgSymbol *s)
{
    if (!ATTR_NODE(s))
        return(NULL);
    return(GRAPHNODE(s)->symb);
}

int FromOtherFile(SgSymbol *s)
{
    if (!ATTR_NODE(s))
       return(1);
    graph_node *gnode = GRAPHNODE(s);
    if(!gnode->st_header || current_file_id != gnode->file_id)
       return(1);
    else
       return(0);
} 

int IsInternalProcedure(SgSymbol *s)
{
    if (!ATTR_NODE(s))
        return 0;
    graph_node *gnode = GRAPHNODE(s);
    if(gnode->st_header &&  gnode->st_header->controlParent()->variant() != GLOBAL && gnode->st_header->controlParent()->variant() != MODULE_STMT)
        return 1;
    else
        return 0;
}

SgStatement *hasInterface(SgSymbol *s)
{
      return (ATTR_NODE(s) ? GRAPHNODE(s)->st_interface : NULL);
}

void SaveInterface(SgSymbol *s, SgStatement *interface)
{
    if (ATTR_NODE(s) &&  !GRAPHNODE(s)->st_interface)
        GRAPHNODE(s)->st_interface = interface; 
}

SgStatement *Interface(SgSymbol *s)
{
    SgStatement *interface = hasInterface(s); 
    if (!interface)    
        interface = getInterface(s);
    
    if (isForCudaRegion() && interface)
    { 
        SaveInterface(s,interface); 
        MarkAsUserProcedure(s);
    }     
    return interface;
}

int findParameterNumber(SgSymbol *s, char *name) 
{
    int i;
    int n = ((SgFunctionSymb *) s)->numberOfParameters();
    for(i=0; i<n; i++)
        if(!strcmp(((SgFunctionSymb *) s)->parameter(i)->identifier(), name))
            return i;
    return -1;
}

int isInParameter(SgSymbol *s, int i)
{
    return (s && ((SgFunctionSymb *) s)->parameter(i) && (((SgFunctionSymb *) s)->parameter(i)->attributes() & IN_BIT) ? 1 : 0);
}
 
SgSymbol *ProcedureSymbol(SgSymbol *s)
{
    if (FromOtherFile(s)) 
    {  
        SgStatement *header = Interface(s);
        return( header ? header->symbol() : NULL);
    }
    return (GetProcedureHeaderSymbol(s));
}

int IsPureProcedure(SgSymbol *s)
{
    SgSymbol *sproc = ProcedureSymbol(s);
    return ( sproc ? sproc->attributes() & PURE_BIT : 0 );
}

int IsElementalProcedure(SgSymbol *s)
{
    SgSymbol *shedr;
    shedr = GetProcedureHeaderSymbol(s);
    if (shedr)
        return(shedr->attributes() & ELEMENTAL_BIT);
    else
        return 0;
}

int IsRecursiveProcedure(SgSymbol *s)
{
    SgSymbol *shedr;
    shedr = GetProcedureHeaderSymbol(s);
    if (shedr)
        return(shedr->attributes() & RECURSIVE_BIT);
    else
        return 0;
}

int isUserFunction(SgSymbol *s)
{
    return(s->attributes() & USER_PROCEDURE_BIT);
}

int IsNoBodyProcedure(SgSymbol *s)
{ 
    if (!ATTR_NODE(s))
        return 0;
    return(GRAPHNODE(s)->st_header == NULL);
}

void MarkAsRoutine(SgSymbol *s)
{
    graph_node *gnode;

    if (!ATTR_NODE(s))
        return;
    gnode = GRAPHNODE(s);
    gnode->is_routine = 1;
    return;
}

void MarkAsCalled(SgSymbol *s)
{
    graph_node *gnode;
    edge *gedge;
    if (!ATTR_NODE(s))
        return;
    gnode = GRAPHNODE(s);
    //if (gnode->st_header)   //  for nobody procedure (for intrinsic functions and ...) gnode->st_header== NULL
    gnode->count++;
    for (gedge = gnode->to_called; gedge; gedge = gedge->next)
        MarkAsCalled(gedge->to->symb);
    return;

}

void MakeFunctionCopy(SgSymbol *s)
{
    SgSymbol *s_header;
    graph_node *gnode;

    if (!ATTR_NODE(s))
        return;
    GRAPHNODE(s)->count++;


    gnode = GRAPHNODE(s);
    s_header = gnode->symb;
    gnode->count++;

    /*
      if(!gnode->st_copy)
      {   printf("make copy of %s\n",s_header->identifier());
      gnode->st_copy = s_header->copySubprogram(*mod_gpu->lexNext()).body();
      }
      */
    //s_copy = &s_header->copySubprogram(*mod_gpu);  *mod_gpu->lexNext()
    //gnode->st_copy = s_header->copySubprogram(*mod_gpu).body(); 
    //gnode->st_copy->unparsestdout();
    //HeaderStatement(&s_header->copySubprogram(*mod_gpu)); //(s_copy); //(s_header->copySubprogram(*mod_gpu)); 
}

SgStatement *HeaderStatement(SgSymbol *s)
{
    return(s->body());
}


void InsertCalledProcedureCopies()
{
    graph_node *ndl;
    int n = 0;
    if (!mod_gpu)
        return;

    SgStatement *after = mod_gpu->lexNext();
    SgStatement *first_kernel_const = after->lexNext();

    for (ndl = node_list; ndl; ndl = ndl->next)
        if (ndl->count)
        {
            if (ndl->st_header && current_file_id == ndl->file_id)  //procedure from current_file
            {
                ndl->st_copy = InsertProcedureCopy(ndl->st_header, ndl->st_header->symbol(), ndl->is_routine, after); //C_Cuda ? mod_gpu : mod_gpu->lexNext());
                n++;
            }
            else     //procedure from other file
                PrototypeOfFunctionFromOtherFile(ndl,after);

            ndl->count = 0;
            ndl->st_interface = NULL;
            //ndl->st_copy = NULL;
        }
       
    if (options.isOn(C_CUDA) && mod_gpu->lexNext()->variant() == COMMENT_STAT)
        mod_gpu->lexNext()->extractStmt(); //extracting empty statement (COMMENT_STAT) 

    if (options.isOn(RTC) && options.isOn(C_CUDA) && n != 0)
        ACC_RTC_AddFunctionsToKernelConsts(first_kernel_const);
    cuda_functions = n;
}

SgSymbol* getReturnSymbol(SgStatement *st_header, SgSymbol *s)
{
    if (st_header->expr(0) == NULL)
        return s;
    else
        return st_header->expr(0)->symbol();    
}

void replaceAttribute(SgStatement *header)
{
        SgExpression *e = new SgExpression(ACC_ATTRIBUTES_OP, new SgExpression(ACC_DEVICE_OP), NULL, NULL);
        header->setExpression(2, *e);
}

int isInterfaceStatement(SgStatement *stmt)
{
    if (stmt->variant() == INTERFACE_STMT || stmt->variant() == INTERFACE_ASSIGNMENT || stmt->variant() == INTERFACE_OPERATOR)
	return 1; 
    return 0;	
}

void ReplaceInterfaceBlocks(SgStatement *header)
{
    SgStatement *last = header->lastNodeOfStmt();
    SgStatement *stmt;
    for (stmt=header->lexNext(); stmt && stmt!=last; stmt=stmt->lexNext())
    {   
        if(isSgExecutableStatement(stmt))
            return;
        if(stmt->variant() == INTERFACE_STMT || stmt->variant() == INTERFACE_ASSIGNMENT || stmt->variant() == INTERFACE_OPERATOR)
	{   
            SgStatement *st_end = stmt->lastNodeOfStmt(); // END INTERFACE
            stmt = stmt->lexNext();
            while(stmt!=st_end)
            {
               if(stmt->variant() == FUNC_HEDR || stmt->variant() == PROC_HEDR )
               {
                   replaceAttribute(stmt);
                   stmt = stmt->lastNodeOfStmt()->lexNext();
               }
               else
                   stmt = stmt->lexNext();
            }                                    
        }           
    } 
}


int HasDerivedTypeVariables(SgStatement *header)
{
    SgSymbol *s;
    SgSymbol *s_last = LastSymbolOfFunction(header);
    
    for (s = header->symbol()->next(); s != s_last->next(); s = s->next())
    {                  
        if( s->type() && s->type()->variant()==T_DERIVED_TYPE)     
        {  // !!! not implemented
           err_p("Derived type variables", header->symbol()->identifier(), 999);
           return 1;
        }
    }
    return 0;
}

SgStatement *InsertProcedureCopy(SgStatement *st_header, SgSymbol *sproc, int is_routine, SgStatement *after)
{
    //insert copy of procedure after statement 'after'
    SgStatement *new_header, *end_st;
    
    SgSymbol *new_sproc = &sproc->copySubprogram(*after);
    new_header = after->lexNext(); // new procedure header  //new_sproc->body()
    SYMB_SCOPE(new_sproc->thesymb) = mod_gpu->thebif;
    new_header->setControlParent(mod_gpu);
    SgSymbol *returnSymbol = getReturnSymbol(new_header, new_sproc);

    if (options.isOn(C_CUDA))
    {
        RenamingNewProcedureVariables(new_sproc); // to avoid conflicts with C language keywords
        int flagHasDerivedTypeVariables = HasDerivedTypeVariables(new_header); 
       
        end_st = new_header->lastNodeOfStmt();
        ConvertArrayReferences(new_header->lexNext(), end_st);  //!!!! 

        TranslateProcedureHeader_To_C(new_header);

        private_list = NULL;

        ExtractDeclarationStatements(new_header);
        SgSymbol *s_last = LastSymbolOfFunction(new_header);
        if (sproc->variant() == FUNCTION_NAME)
        {  
            SgSymbol *sfun = &new_sproc->copy();
            new_header->expr(0)->setSymbol(sfun); //fe->setSymbol(sfun);
            SYMB_IDENT(new_sproc->thesymb) = FunctionResultIdentifier(new_sproc);
           
            InsertReturnBeforeEnd(new_header, end_st);
        }

        swapDimentionsInprivateList();
        std::vector < std::stack < SgStatement*> > zero = std::vector < std::stack < SgStatement*> >(0);
        cur_func = after;
        Translate_Fortran_To_C(new_header, end_st, zero, 0);   //TranslateProcedure_Fortran_To_C(after->lexNext());
         
        if (sproc->variant() == FUNCTION_NAME)
        {   
            new_header->insertStmtAfter(*Declaration_Statement(new_sproc), *new_header);
            ChangeReturnStmts(new_header, end_st, returnSymbol);
        }
        if(!flagHasDerivedTypeVariables) //!!! derived data type is not supported          
            MakeFunctionDeclarations(new_header, s_last);
        
        newVars.clear();
        private_list = NULL;
        // generate prototype of function and insert it before 'after'
        if (options.isOn(RTC) == false)
            doPrototype(new_header, mod_gpu, is_routine ? !STATIC : STATIC);

    }
    else       //Fortran Cuda
    {
        replaceAttribute(new_header);
        new_header->addComment("\n");  // add comment (empty line) to new procedure header
        ReplaceInterfaceBlocks(new_header);
    }
    
    return(new_header);
}

SgStatement *FunctionPrototype(SgSymbol *sf)
{                                  
    SgExpression *fref = new SgFunctionRefExp(*sf);
    fref->setSymbol(*sf);
    fref->setType(*sf->type());
    SgStatement *st = new SgStatement(VAR_DECL);
    st->setExpression(0, *new SgExprListExp(*fref));

    return (st);
}


void doPrototype(SgStatement *func_hedr, SgStatement *block_header, int static_flag)
{
    SgSymbol *sf = func_hedr->expr(0)->symbol();
    SgStatement *st = FunctionPrototype(sf);
    if (func_hedr->expr(0)->lhs())
        st->expr(0)->lhs()->setLhs(func_hedr->expr(0)->lhs()->copy());
    st->addDeclSpec(BIT_CUDA_DEVICE);
    if (static_flag)
        st->addDeclSpec(BIT_STATIC);

    block_header->insertStmtAfter(*st, *block_header);        //before->insertStmtAfter(*st,*before->controlParent());
}

SgStatement  *TranslateProcedureHeader_To_C(SgStatement *new_header)
{
    SgSymbol *new_sproc =  new_header->symbol();
    SgFunctionRefExp *fe = new SgFunctionRefExp(*new_sproc);
    fe->setSymbol(*new_sproc);
    new_header->setExpression(0, *fe);
    SgSymbol *returnSymbol = getReturnSymbol(new_header, new_sproc);
    if (new_sproc->variant() == PROCEDURE_NAME)
        new_sproc->setType(C_VoidType());
    else                     // FUNCTION_NAME
    {
       //new_sproc->setType(C_Type(new_sproc->type()));
        new_sproc->setType(C_Type(returnSymbol->type()));
    }
    fe->setType(new_sproc->type());
    fe->setLhs(FunctionDummyList(new_sproc));
    BIF_LL3(new_header->thebif) = NULL; 
    new_header->addDeclSpec(BIT_CUDA_DEVICE);
    new_header->setVariant(FUNC_HEDR); 
    return new_header;
}

void PrototypeOfFunctionFromOtherFile(graph_node *node, SgStatement *after)
{
    if (options.isOn(RTC)) return;
    if(!node->st_interface) return;

    SgStatement *interface = node->st_interface;
    //SgSymbol *sproc = interface->symbol()
    //SgSymbol *new_sproc = new SgSymbol(sproc->variant(), sproc->identifier(), sproc->type(), current_file->firstStatement(),);

    SgSymbol *sh = &(interface->symbol()->copyLevel1());     
    SYMB_SCOPE(sh->thesymb) = current_file->firstStatement()->thebif;
    SgStatement *new_hedr = &(interface->copy()); 
    new_hedr->setSymbol(*sh);
    TranslateProcedureHeader_To_C(new_hedr);
    doPrototype(new_hedr, mod_gpu, !STATIC);
    
    //current_file->firstStatement()->insertStmtAfter(*new_hedr, *current_file->firstStatement()); 
    //SYMB_FUNC_HEDR(sh->thesymb) = new_hedr->thebif; 


    //node->st_interface->setLexNext(*node->st_interface->lastNodeOfStmt());
    //SgStatement *hedr_st = InsertProcedureCopy(node->st_interface, node->st_interface->symbol(), after);
    //hedr_st->extractStmt();
    node->st_interface = NULL;
    return;
}

SgExpression *FunctionDummyList(SgSymbol *s)
{
    SgExpression *arg_list = NULL, *ae = NULL;

    int n = ((SgFunctionSymb *)s)->numberOfParameters();
    
    //insert at 0-th position inf-argument
    //check for optional arguments, if some argunemt exist with optional then add argument-mask
    
    //int useOption = false;
    //for (i = 0; i < n; i++)
    //{
    //    useOption |= ((SgFunctionSymb *)s)->parameter(i)->attributes() & OPTIONAL_BIT;
    //}
    //if(useOption)
    //{
    //    std::string nameForArgsInfo = "arg_info";   // name for new arguments 
    //    SgSymbol* argInfo = new SgSymbol(VARIABLE_NAME,nameForArgsInfo.c_str());
    //    argInfo->setType(C_LongType());
    //    ae = new SgVarRefExp(argInfo);
    //    ae = new SgExprListExp(*ae);
    //    arg_list = AddListToList(arg_list, ae);
    //}

    for (int i = 0; i < n; i++)
    {
        SgSymbol *sarg = ((SgFunctionSymb *)s)->parameter(i);

        if (!isSgArrayType(sarg->type()))
        {
            sarg->setType(C_Type(sarg->type()));
            if (sarg->attributes() & OPTIONAL_BIT)
            {
                sarg->setType(new SgDerivedTemplateType(new SgTypeRefExp(*sarg->type()), new SgSymbol(TYPE_NAME, "optArg")));
            }
            ae = new SgVarRefExp(sarg);
            //ae->setType(C_ReferenceType(sarg->type()));
            if (sarg->attributes() & IN_BIT)
                ae = new SgExprListExp(*ae);
            else
                ae = new SgExprListExp(SgAddrOp(*ae));
            arg_list = AddListToList(arg_list, ae);

        }
        else
        {
            int needChanged = true;
            SgArrayType* arrT = (SgArrayType*)sarg->type();
            int dims = arrT->dimension();
            SgExpression *dimList = arrT->getDimList();
            
            while (dimList)
            {
                if (dimList->lhs()->variant() != DDOT)
                {
                    needChanged = false;
                    break;
                }
                else if (dimList->lhs()->rhs())
                {
                    needChanged = false;
                    break;
                }
                dimList = dimList->rhs();
            }

            SgType *t = C_PointerType(C_Type(sarg->type()->baseType()));
            sarg->setType(t);
            ae = new SgVarRefExp(sarg);
            ae->setType(t);
            if (needChanged)
            {
                sarg->setType(new SgDerivedTemplateType(new SgTypeRefExp(*t), new SgSymbol(TYPE_NAME, "s_array")));
                ae = new SgVarRefExp(sarg);
                ae = new SgExprListExp(*ae);
                arg_list = AddListToList(arg_list, ae);
                continue;
            }

            //ae->setType(C_ReferenceType(sarg->type())); 
            ae = new SgExprListExp(*new SgPointerDerefExp(*ae));
            arg_list = AddListToList(arg_list, ae);
            //SgSymbol *arr_info = new SgSymbol(VAR_REF, ("inf_" + std::string(sarg->identifier())).c_str());
            //arr_info->setType(C_PointerType(C_Type(new SgType(T_INT))));
            //ae = new SgVarRefExp(arr_info);
            //ae = new SgExprListExp(*new SgPointerDerefExp(*ae));
            //arg_list = AddListToList(arg_list, ae);
        }
    }
    return (arg_list);
}

char *FunctionResultIdentifier(SgSymbol *sfun)
{
    char *name;
    name = (char *)malloc((unsigned)(strlen(sfun->identifier()) + 4 + 1));
    sprintf(name, "%s_res", sfun->identifier());
    return(NameCheck(name, sfun));
}

SgSymbol *isSameNameInProcedure(char *name, SgSymbol *sfun)
{
    SgSymbol *s;
    for (s = sfun->next(); s; s = s->next())
    if (!strcmp(s->identifier(), name))
        return(s);
    return(NULL);
}

char *NameCheck(char *name, SgSymbol *sfun)
{
    SgSymbol *s;
    while ((s = isSameNameInProcedure(name, sfun)) != 0)
    {
        name = (char *)malloc((unsigned)(strlen(name) + 2));
        sprintf(name, "%s_", s->identifier());
    }
    return(name);
}

void InsertReturnBeforeEnd(SgStatement *new_header, SgStatement *end_st)
{
    SgStatement *prev = end_st->lexPrev();
    if (prev->variant() == RETURN_STAT)
        return;
    prev->insertStmtAfter(*new SgStatement(RETURN_STAT), *new_header);
}

void ChangeReturnStmts(SgStatement *new_header, SgStatement *end_st, SgSymbol *sres)
{
    SgStatement *stmt;
    for (stmt = new_header->lexNext(); stmt != end_st; stmt = stmt->lexNext())
    if (stmt->variant() == RETURN_STAT)
        stmt->setExpression(0, *new SgVarRefExp(sres));

}

template<typename callStatType>
static void createIntefacePrototype(callStatType *funcDecl)
{
    string funcName = funcDecl->name().identifier();
    const int parNum = funcDecl->numberOfParameters();
    vector<SgType*> prototype(parNum);
    for (int i = 0; i < parNum; ++i)
    {
        SgSymbol *par = funcDecl->parameter(i);
        SgType *type = par->type();
        prototype[i] = type;
    }
    map <string, vector<vector<SgType*> > >::iterator it = interfaceProcedures.find(funcName);
    if (it == interfaceProcedures.end())
    {
        vector<vector<SgType*> > prototypes = vector<vector<SgType*> >();
        prototypes.push_back(prototype);

        interfaceProcedures.insert(it, make_pair(funcName, prototypes));
    }
    else
        it->second.push_back(prototype);
}

bool CreateIntefacePrototype(SgStatement *header)
{
    bool retVal = true;
    if (header->variant() == FUNC_HEDR)
    {        
        SgFuncHedrStmt *funcDecl = isSgFuncHedrStmt(header);        
        if (funcDecl)
            createIntefacePrototype(funcDecl);
        else
            retVal = false;
    }
    else if (header->variant() == PROC_HEDR)
    {
        SgProcHedrStmt *procDecl = isSgProcHedrStmt(header);
        if (procDecl)
            createIntefacePrototype(procDecl);
        else
            retVal = false;
    }
    else
        retVal = false;

    return retVal;
}

void ExtractDeclarationStatements(SgStatement *header)
{
    SgStatement *cur_st;
    SgStatement *stmt = header->lexNext();
    SgExprListExp *e;
    SgExpression *list, *it;

    if(stmt->variant()==CONTROL_END)
        return;

    while (stmt && !isSgExecutableStatement(stmt)) //is Fortran specification statement
    {   
        cur_st = stmt;     
        stmt = stmt->lexNext();
        if(cur_st->variant() == INTERFACE_STMT || cur_st->variant() == INTERFACE_ASSIGNMENT || cur_st->variant() == INTERFACE_OPERATOR)
        {
            SgStatement *last = cur_st->lastNodeOfStmt();
            SgStatement *start = cur_st;
            while (start != last)
            {
                // save prototypes of FUNC and PROC 
                if (start->variant() == FUNC_HEDR)
                {
                    SgFuncHedrStmt *funcDecl = isSgFuncHedrStmt(start);
                    if (funcDecl)
                    {
                        createIntefacePrototype(funcDecl);
                        start = funcDecl->lastNodeOfStmt();
                    }
                }
                else if (start->variant() == PROC_HEDR)
                {
                    SgProcHedrStmt *procDecl = isSgProcHedrStmt(start);
                    if (procDecl)
                    {
                        createIntefacePrototype(procDecl);
                        start = procDecl->lastNodeOfStmt();
                    }
                }
                start = start->lexNext();
            }
            stmt = cur_st->lastNodeOfStmt()->lexNext();
            cur_st->extractStmt();
            continue;
        }
        if(cur_st->variant()==STRUCT_DECL) 
        {
            stmt = cur_st->lastNodeOfStmt()->lexNext();
            cur_st->extractStmt();
            continue;
        }
        //if(cur_st->variant()==IMPL_DECL || cur_st->variant()==DATA_DECL || cur_st->variant()==USE_STMT || cur_st->variant()==FORMAT_STAT || cur_st->variant()==ENTRY_STAT || cur_st->variant()==COMM_STAT || cur_st->variant()==STMTFN_STAT )
        if(!isSgVarDeclStmt(cur_st) && !isSgVarListDeclStmt(cur_st))
        {  
            cur_st->extractStmt(); 
            continue;     
        }

        list = cur_st->expr(0);
        for(; list; list = list->rhs())
        {
            if(IS_DUMMY(list->lhs()->symbol()) || !isSgArrayType(list->lhs()->symbol()->type()))
                continue;
            //add local array in private list
            e = new SgExprListExp(*new SgVarRefExp(*list->lhs()->symbol()));
            e->setRhs(private_list);
            private_list = e;
        }        
        cur_st->extractStmt();
    }
}

/*
std::string ArrParametrs(SgSymbol* arr)
{
    return ("inf_" + std::string(arr->identifier())).c_str();
}
SgExpression* InheritUpperBound(SgSymbol* arr, int i)
{
    SgExpression *dim = ((SgArrayType *)(arr->type()))->sizeInDim(i);
    SgExpression *lb = dim->lhs();
    SgExpression *ub = dim->rhs();
    if(dim->variant() != DDOT || ub != NULL)
    {
        return UpperBound(arr,i);
    }
    if(lb == NULL)
    {
        return  &(*(new SgArrayRefExp(*new SgSymbol(VARIABLE_NAME, ArrParametrs(arr).c_str()), *new SgValueExp((i-1)+7)))
            - *(new SgArrayRefExp(*new SgSymbol(VARIABLE_NAME,  ArrParametrs(arr).c_str()), *new SgValueExp(i-1)))
            + *new SgValueExp(1)) ; 
    }
    else if(1)
    {
        return  &(*(new SgArrayRefExp(*new SgSymbol(VARIABLE_NAME, ArrParametrs(arr).c_str()), *new SgValueExp((i-1)+7)))
            - *(new SgArrayRefExp(*new SgSymbol(VARIABLE_NAME,  ArrParametrs(arr).c_str()), *new SgValueExp(i-1)))
            + *lb) ;
    }

}
SgExpression* InheritLowerBound(SgSymbol* arr, int i)
{
    SgExpression *dim = ((SgArrayType *)(arr->type()))->sizeInDim(i);
    SgExpression *lb = dim->lhs();
    SgExpression *ub = dim->rhs();
    if(dim->variant() != DDOT || ub != NULL)
    {
        return UpperBound(arr,i);
    }
    if(lb == NULL)
    {
        return new SgValueExp(1) ; 
    }
    else
    {
        return lb;
    }

}
*/
void CorrectSubscript(SgExpression *e)
{
    int dims = ((SgArrayType *)(e->symbol()->type()))->dimension();
    std::deque<std::pair<SgExpression*, SgExpression*> > koefs;
//    SgExpression *infUpperBound = NULL;                ;
//    SgExpression *infLowerBound = NULL;
    SgExpression *tmp = e->lhs();
    if (tmp == NULL)
    {
        return;
    }
    for (int i = 0; i < dims; ++i)
    {
        SgExpression *dimsize = ((SgArrayType *)(e->symbol()->type()))->sizeInDim(i);
        if (dimsize->variant() == STAR_RANGE)
        {
            break;
        }
    }
    for (int i = 0; i < dims; ++i)
    {
        std::pair<SgExpression*, SgExpression*> tmp_pair;
        SgExpression * koef = new SgValueExp(1);
        SgExpression *dimsize = ((SgArrayType *)(e->symbol()->type()))->sizeInDim(i);
        SgExpression *check = dimsize->lhs();
        for (int j = 0; j < i; ++j)
        {
//            SgExpression *dimsize = ((SgArrayType *)(e->symbol()->type()))->sizeInDim(j);
//            if (isSgSubscriptExp(dimsize) && !dimsize->rhs())
//            {
//                infLowerBound = (new SgArrayRefExp(*new SgSymbol(VARIABLE_NAME, ArrParametrs(e->symbol()).c_str()), *new SgValueExp(j)));
//                infUpperBound = (new SgArrayRefExp(*new SgSymbol(VARIABLE_NAME, ArrParametrs(e->symbol()).c_str()), *new SgValueExp(j+7)));
//
//                koef = Calculate(&(*koef * (*infUpperBound - *infLowerBound + *new SgValueExp(1))));
//
//            }
//            else
//            {
            SgExpression * up = UpperBound(e->symbol(), j);
            if(up->variant() == FUNC_CALL)
            {
                up = new SgExpression(RECORD_REF);
                up->setLhs(new SgVarRefExp(e->symbol()));
                //up->setRhs(new SgVarRefExp(*new SgSymbol(FIELD_NAME,(std::string("ub[")+std::to_string(j)+std::string("]")).c_str())));
                up->setRhs(new SgFunctionCallExp(*new SgSymbol(MEMBER_FUNC,"ub"), *new SgExprListExp(*new SgValueExp(j))));
            }
            SgExpression * low = LowerBound(e->symbol(), j);
            koef = Calculate(&(*koef * (*up - *LowerBound(e->symbol(), j) + *new SgValueExp(1))));
//            }
        }
        tmp_pair.first = koef;

        tmp_pair.second = Calculate(&(*tmp->lhs() - *LowerBound(e->symbol(), i)));
        tmp = tmp->rhs();
        koefs.push_back(tmp_pair);

    }
    SgExpression *line = koefs.front().second;
    koefs.pop_front();
    tmp = e->lhs();
    for (int i = 0; i < dims - 1; ++i)
    {
        line = &(*koefs.front().second * *koefs.front().first + *line);
        koefs.pop_front();
        tmp = tmp->rhs();
    }
    e->setLhs((new SgExprListExp(*line)));
}

void replaceVectorRef(SgExpression *e)
{
    SgType *type;
    if (e == NULL)
        return;
    if (isSgArrayRefExp(e))
    {
        type = isSgArrayType(e->symbol()->type());
        if (IS_DUMMY(e->symbol()) && type)
        {
            CorrectSubscript(e);
        }
        return;
    }

    replaceVectorRef(e->lhs());
    replaceVectorRef(e->rhs());
}

void ConvertArrayReferences(SgStatement *first, SgStatement *last)
{
    SgStatement *st;
    for (st = first; st != last; st = st->lexNext())
    {
        if (st->expr(0))
            replaceVectorRef(st->expr(0));
        if (st->expr(1))
            replaceVectorRef(st->expr(1));
        if (st->expr(2))
            replaceVectorRef(st->expr(2));
    }
}

void convertArrayDecl(SgSymbol* s)
{
    SgExprListExp *resDims, *tmp; 
    std::stack<SgExpression*>dims;
    if(isSgArrayType(s->type()))    
    {
        SgExpression *dimList = isSgArrayType(s->type())->getDimList();
        while (dimList)
        {
            if(dimList->lhs()->variant() == DDOT)
            {
                dims.push(Calculate(&(*(dimList->lhs()->rhs()) - *(dimList->lhs()->lhs()) + *new SgValueExp(1))));
            }
            else
            {       
                dims.push(Calculate(&(*(dimList->lhs()))));
            }
            dimList = dimList->rhs();
        }
        SgType* t = C_Type(isSgArrayType(s->type())->baseType());
        SgArrayType *arr = new SgArrayType(*t); 
        while (!dims.empty())
        {
            arr->addDimension(dims.top());
            dims.pop();
        }
        s->setType(arr);
    }


}

void MakeFunctionDeclarations(SgStatement *header, SgSymbol *s_last)
{
    SgSymbol *s;
    SgStatement *cur_stat = header;
    SgStatement *st;
    SgExpression *el;
    char* name = header->expr(0)->symbol()->identifier();

    for (s = header->symbol()->next(); s != s_last->next(); s = s->next())
    {
        if (isSgFunctionSymb(s) != NULL)
            continue;

        int flags = s->attributes();

        if (IS_DUMMY(s))
        {
            if (flags & (IN_BIT | OUT_BIT | INOUT_BIT))
                ;
            else if(!options.isOn(NO_PURE_FUNC))
                err_p("Dummy argument need to have INTENT attribute in PURE procedure", name, 617);
            continue;
        }

        if (flags & SAVE_BIT)
            err_p("SAVE not be used in PURE procedure", name, 618);
        if (flags & COMMON_BIT)
            err_p("COMMON not be used in PURE procedure", name, 619);

        if (s->scope() != header)
        {    
            //printf("%s: %d \n",s->identifier(),s->scope()->variant());  //printf("%s: %d %s \n",s->identifier(),s->scope()->variant(),s->scope()->symbol()->identifier());
            continue;
        }
        if (!isSgArrayType(s->type()))  //scalar variable
            s->setType(C_Type(s->type()));
        else
        {
            continue;
        }

        if (isSgConstantSymb(s))
        {   
            SgExpression *ce = ((SgConstantSymb *)s)->constantValue();            
            convertExpr(ce, ce);
            st = makeSymbolDeclarationWithInit(s, ce);
            st->addDeclSpec(BIT_CONST);
        }
        else  if(isSgVariableSymb(s))
            st = makeSymbolDeclaration(s);         //st = Declaration_Statement(s);
        else
            continue;
        cur_stat->insertStmtAfter(*st);
        cur_stat = st;
    }
                //printf("\n"); if(private_list) private_list->unparsestdout(); printf("\n");  
    for (el = private_list; el; el = el->rhs())
    {
        convertArrayDecl(el->lhs()->symbol());
        st = makeSymbolDeclaration(el->lhs()->symbol());
        cur_stat->insertStmtAfter(*st);
        cur_stat = st;
    }
}

SgSymbol *LastSymbolOfFunction(SgStatement *header)
{
    SgSymbol *s = header->symbol();
    while (s->next())
    {   //printf("       %s: %d %s\n", s->next()->identifier(),s->next()->scope()->variant(), s->next()->scope()->symbol() ? s->next()->scope()->symbol()->identifier() : "N");
        s = s->next();
    }
    return(s);
}


//---------------------------------------------------------------------------------------
void ProjectStructure(SgProject &project)
{
    int n = project.numberOfFiles();
    SgFile *file;
    int i;
    // building program structure 
    // looking through the file list of project (first time)
    for (i = n - 1; i >= 0; i--)
    {
        file = &(project.file(i));
        current_file = file;
        current_file_id = i;
        FileStructure(file);
        //printf("%s %d\n",project.fileName(i),i);  PrintWholeGraph();   
    }
    for (i = n - 1; i >= 0; i--)
    {
        file = &(project.file(i));
        current_file = file;
        current_file_id = i;
        doCallGraph(file);
    }
    //ScanSymbolTable(file); 
    //PrintWholeGraph();
}

void FileStructure(SgFile *file)
{// looking through the file and creating graph node for header of each program unit 
    SgStatement *stat;

    // grab the first statement in the file.
    stat = file->firstStatement(); // file header   
    for (stat = stat->lexNext(); stat; stat = stat->lexNext())
    {
        if (stat->variant() == INTERFACE_STMT || stat->variant() == INTERFACE_ASSIGNMENT || stat->variant() == INTERFACE_OPERATOR)
        {
            stat = stat->lastNodeOfStmt(); //InterfaceBlock(stat);  
            continue;
        }

        if (stat->variant() == FUNC_HEDR || stat->variant() == PROC_HEDR || stat->variant() == PROG_HEDR || stat->variant() == MODULE_STMT)
        {                                      //printf("%d %s \n",stat->lineNumber(),stat->symbol()->identifier());
            //creating graph node for header of function (procedure, program)
            cur_node = GraphNode(stat->symbol(), stat, NEW);

        }

    }

}

void ReplaceGenericInterfaceBlocks(SgStatement *hedr, SgStatement *end_of_unit)
{
    SgStatement *stmt;
    //SgSymbol *symb = NULL;
    for (stmt = hedr->lexNext(); stmt != end_of_unit; stmt = stmt->lastNodeOfStmt()->lexNext())
    {
        if(stmt->variant() == INTERFACE_STMT && stmt->symbol())
            BIF_SYMB(stmt->thebif) = NULL; 
        if(stmt->variant() == FUNC_HEDR || stmt->variant() == PROC_HEDR )
            stmt = stmt->lexNext();
    }
}


void doCallGraph(SgFile *file)
{// scanning the file to search procedure calls
    SgStatement *stat = NULL, *end_of_unit = NULL;
    //char *func_name;  
    //int *ir;
    //int has_main_program_unit = 0;

    // grab the first statement in the file.
    stat = file->firstStatement(); // file header  
	for (stat = stat->lexNext(); stat; stat = end_of_unit->lexNext())
        {
             has_generic_interface = 0;
	     end_of_unit = ProgramUnit(stat);
             if (has_generic_interface)
                 ReplaceGenericInterfaceBlocks(stat,end_of_unit);
        }
    // add the attribute (last statement of file) to first statement of file
    SgStatement **last = new (SgStatement *);
#if __SPF
    addToCollection(__LINE__, __FILE__, last, 1);
#endif
    *last = end_of_unit;
    file->firstStatement()->addAttribute(LAST_STATEMENT, (void*)last, sizeof(SgStatement *));

}

SgStatement *ProgramUnit(SgStatement *first)
{
    SgStatement *stat, *end_of_unit;

    // program unit: main program, external subprogram, module or block data
    for (stat = first; stat; stat = end_of_unit->lexNext())
    {
        //end of program unit with CONTAINS statement
        if (stat->variant() == CONTROL_END)
        {
            if (stat->controlParent() == first)  //end of program unit with CONTAINS statement
                return(stat);
            else
            {
                end_of_unit = stat;
                continue;
            }
        }
        if (stat->variant() == BLOCK_DATA) //BLOCK_DATA header 
            return(stat->lastNodeOfStmt());

        // PROGRAM, SUBROUTINE, FUNCTION  or MODULE header

        //scanning the Symbols Table of the function 
        //     ScanSymbTable(func->symbol(), (f->functions(i+1))->symbol());

        end_of_unit = Subprogram(stat);  // end_of unit may be END or CONTAINS statement
        //printf("---%d  %d %s \n",stat->lineNumber(),end_of_unit->lineNumber(),stat->symbol()->identifier());
        GRAPHNODE(stat->symbol())->st_last = end_of_unit;
        if (end_of_unit->variant() == CONTROL_END && end_of_unit->controlParent() == first) //end of program unit without CONTAINS statement  
            return(end_of_unit);
    }
    return NULL;
}

SgStatement *Subprogram(SgStatement *func)
{
    // Build a directed acyclic call multigrahp (call DAMG) 
    // which represents calls between routines of the program

    SgStatement *stmt, *last, *first;


    DECL(func->symbol()) = 1;
    HEDR(func->symbol()) = func->thebif;
    cur_func = func;
    //if( func->variant() == PROG_HEDR)
    //   PROGRAM_HEADER(func->symbol()) = func->thebif;

    // determing graph node for header of function (procedure, program)
    cur_node = ATTR_NODE(func->symbol()) ? GRAPHNODE(func->symbol()) : GraphNode(func->symbol(), func, 0);

    first = func->lexNext();
    //printf("\n%s  header_id= %d \n", func->symbol()->identifier(), func->symbol()->id());
    //!!!debug
    //if(fsymb)
    //printf("\n%s   %s \n", header(func->variant()),fsymb->identifier()); 
    //else {
    //printf("Function name error  \n");
    //return;
    //}

    last = func->lastNodeOfStmt();

    // follow the statements of the function in lexical order
    // until last statement
    for (stmt = first; stmt && (stmt != last); stmt = stmt->lexNext())
    {   
        switch (stmt->variant()) {

        case CONTAINS_STMT:
            last = stmt;
            goto END_;
            break;

        case ENTRY_STAT:
            // !!!!!!!
            break;

        case DATA_DECL:
        case CONTROL_END:
        case STOP_STAT:
        case PAUSE_NODE:
        case GOTO_NODE:             // GO TO             
            break;

        case VAR_DECL:
        case SWITCH_NODE:           // SELECT CASE ...
        case ARITHIF_NODE:          // Arithmetical IF
        case IF_NODE:               // IF... THEN
        case WHILE_NODE:            // DO WHILE (...) 
        case CASE_NODE:             // CASE ...
        case ELSEIF_NODE:           // ELSE IF...
        case LOGIF_NODE:            // Logical IF
            FunctionCallSearch(stmt->expr(0));
            break;
        case STMTFN_STAT:
            DECL(stmt->expr(0)->symbol()) = 2;
            break;
        case COMGOTO_NODE:          // Computed GO TO
        case OPEN_STAT:
        case CLOSE_STAT:
        case INQUIRE_STAT:
        case BACKSPACE_STAT:
        case ENDFILE_STAT:
        case REWIND_STAT:
            FunctionCallSearch(stmt->expr(1));
            break;

        case PROC_STAT:  {           // CALL
                             SgExpression *el;
                             int inlined;
                             //printf("\n%s  call_id= %d \n", stmt->symbol()->identifier(), stmt->symbol()->id());
                             //!!!temporary
                             //inlined = (func->variant() == PROG_HEDR) ? 0 : 1;
                             inlined = 1;
                             Call_Site(stmt->symbol(), inlined, stmt, NULL);
                             // looking through the arguments list
                             for (el = stmt->expr(0); el; el = el->rhs())
                                 Arg_FunctionCallSearch(el->lhs());   // argument
        }
            break;

        case ASSIGN_STAT:             // Assign statement
        case WRITE_STAT:
        case READ_STAT:
        case PRINT_STAT:
        case FOR_NODE:
            FunctionCallSearch(stmt->expr(0));   // left part
            FunctionCallSearch(stmt->expr(1));   // right part
            break;
        case ACC_REGION_DIR:
            in_region++;
            break; 
        case ACC_END_REGION_DIR:
            in_region--;
            break;
        default:
            FunctionCallSearch(stmt->expr(0));
            FunctionCallSearch(stmt->expr(1));
            FunctionCallSearch(stmt->expr(2));
            break;
        }

    } // end of processing statement/directive 

END_:
    // for debugging
    if (deb_reg > 1)
        PrintGraphNode(cur_node);

    return(last);

}

void FunctionCallSearch(SgExpression *e)
{
    SgExpression *el;
    if (!e)
        return;

    if (isSgFunctionCallExp(e)) {
        Call_Site(e->symbol(), 1, NULL, e);
        for (el = e->lhs(); el; el = el->rhs())
            Arg_FunctionCallSearch(el->lhs());
        return;
    }
    FunctionCallSearch(e->lhs());
    FunctionCallSearch(e->rhs());
    return;
}

void Arg_FunctionCallSearch(SgExpression *e)
{
    FunctionCallSearch(e);
    return;
}

void FunctionCallSearch_Left(SgExpression *e)
{
    FunctionCallSearch(e);
}

int isAsterDummy(SgSymbol *s)
{
    if (!s) return 0;
    if (!strcmp(s->identifier(),"*")) return 1;
    return 0;
}

SgExpression * TypeKindExpr(SgType *t)
{
    SgExpression *len;
    SgExpression *selector;
    if(!t) return (NULL);
    len = t->length(); 
    selector = t->selector(); 
    //printf("\nTypeSize");
    //printf("\nranges:"); if(len) len->unparsestdout();
    //printf("\nkind_len:");  if(selector) selector->unparsestdout();

    //the number of bytes is not specified in type declaration statement
    if (!len && !selector) 
        return (new SgValueExp(IntrinsicTypeSize(t)));
    if (t->variant() != T_STRING) // numeric types
    {
        if (len && !selector)   //INTEGER*2,REAL*8,CHARACTER*(N+1)
            return(Calculate(len));
        else
            return(Calculate(selector->lhs() ? selector->lhs() : selector)); //specified kind:INT_VAL for literal constants or KIND_OP 
    }
    else  //  character (T_STRING)
    {
        if (!selector->lhs())  // for literal constants 1_"xxx"
            return(Calculate(selector));
        else if (selector->variant() == KIND_OP)
            return(Calculate(selector->lhs()));
        else if (selector->variant() == LENGTH_OP)
            return(new SgValueExp(IntrinsicTypeSize(t)));
        else if (selector->lhs()->variant()==KIND_OP)   
            return(Calculate(selector->lhs()));
        else if (selector->rhs()->variant()==KIND_OP)   
            return(Calculate(selector->rhs()));
    }
    return (NULL);
}

int CompareKind(SgType *type_arg, SgType *type_dummy)
{  
    int kind1=-1, kind2=-1;
    SgExpression *e1 = TypeKindExpr(type_dummy); 
    if (e1 && e1->isInteger())
        kind1 = e1->valueInteger();

    SgExpression *e2 = TypeKindExpr(type_arg);  
    if (e2 && e2->isInteger())
        kind2 = e2->valueInteger();

    if (kind1>=0 && kind1 == kind2)
        return 1;
    else 
        return 0;
}

int  CompareTypeKindRank (SgExpression *e, SgSymbol *dummy)
{
    if (!dummy) return 0;
    if (e->variant() == ARRAY_OP)
       CompareTypeKindRank (e->lhs(), dummy);
    //if (isSgRecordRefExp(e))
    //    CompareTypeKindRank (RightMostField(e), dummy); 
    if (!e->type() && !dummy->type()) 
         return 1;
    else if (!e->type())
         return 0;
    else if (!dummy->type())
         return 0;

    SgArrayType *artype_dummy = isSgArrayType(dummy->type());
    SgArrayType *artype_arg   = isSgArrayType(e->type());
    if (artype_dummy != 0 && artype_arg != 0)
    {
       if (TYPE_DIM(artype_dummy->thetype) != TYPE_DIM(artype_arg->thetype))  //dimension() method cannot be used
           return 0; 
    } 
    else if (artype_dummy == 0 && artype_arg == 0)
       ;
    else
       return 0;
    SgType *type_arg = artype_arg ? artype_arg->baseType() : e->type();
    SgType *type_dummy = artype_dummy ? artype_dummy->baseType() : dummy->type();
    
    if (type_dummy->variant() == T_DERIVED_TYPE && type_arg->variant() == T_DERIVED_TYPE)
    {
        if (!strcmp(ORIGINAL_SYMBOL(type_dummy->symbol())->identifier(), ORIGINAL_SYMBOL(type_arg->symbol())->identifier())) 
            return 1;
        else
            return 0;
    }
    else if (type_dummy->variant() == T_DERIVED_TYPE || type_arg->variant() == T_DERIVED_TYPE)
        return 0;
    if (type_dummy->variant() == T_STRING)
    { 
        if( type_arg->variant() == T_STRING)
            return 1;
        else
            return 0; 
    }    
    if ( type_dummy->variant() == T_COMPLEX || type_dummy->variant() == T_DCOMPLEX)
        if ( type_arg->variant() == T_COMPLEX || type_arg->variant() == T_DCOMPLEX)
            return (CompareKind(type_arg, type_dummy));
        else
            return 0;
    if (type_dummy->variant() == T_FLOAT || type_dummy->variant() == T_DOUBLE)
        if (type_arg->variant() == T_FLOAT ||   type_arg->variant() == T_DOUBLE)
            return (CompareKind(type_arg,type_dummy));
        else
            return 0;
    if (type_arg->variant() != type_dummy->variant()) 
        return 0;
    
    return (CompareKind(type_arg,type_dummy));
}

int CompareArgDummy(SgExpression *e, int i, SgSymbol *symb)
{    
    if (i == -1) return 0;
    if (e->variant() == KEYWORD_ARG) 
        CompareArgDummy(e->rhs(), findParameterNumber(symb, NODE_STR(e->lhs()->thellnd)), symb);
    //if((((SgFunctionSymb *) symb)->parameter(i))->attributes() & OPTIONAL_BIT ) return 1;
    if (e->variant() == LABEL_ARG) return isAsterDummy(((SgFunctionSymb *) symb)->parameter(i)); //!!! illegal
    return (CompareTypeKindRank(e, ((SgFunctionSymb *) symb)->parameter(i) ));
}

int CompareArguments(SgSymbol *symb, SgExpression *arg_list)
{
    SgExpression *el, *e;
    int i;   
    for (el = arg_list, i = 0; el; el = el->rhs(), i++)
        if (!CompareArgDummy(el->lhs(), i, symb))
            return 0;
    return 1;    
}

SgStatement *getInterfaceInScope(SgSymbol *s, SgStatement *func)
{
    enum { SEARCH_INTERFACE, CHECK_INTERFACE, FIND_NAME };

    SgStatement *searchStmt = func->lexNext();
    SgStatement *tmp;
    const char *funcName = s->identifier();
    const char *toCmp;

    int mode = SEARCH_INTERFACE;
    //search interface in the specification part of a program unit
    while (searchStmt && (!isSgExecutableStatement(searchStmt) || isDvmSpecification(searchStmt)))
    {
        switch (mode)
        {
        case SEARCH_INTERFACE:
            if (searchStmt->variant() != INTERFACE_STMT)
                searchStmt = searchStmt->lastNodeOfStmt()->lexNext();
            else
                mode = CHECK_INTERFACE;
            break;
        case CHECK_INTERFACE:
            if (searchStmt->symbol())
                toCmp = searchStmt->symbol()->identifier();
            else
                toCmp = "";

            if (searchStmt->symbol() && strcmp(toCmp, funcName) != 0)
            {
                searchStmt = searchStmt->lastNodeOfStmt()->lexNext();
                mode = SEARCH_INTERFACE;
            }
            else
            {
                if(searchStmt->symbol())
                {
                    return searchStmt;
                }
                else
                {
                    mode = FIND_NAME;
                    searchStmt = searchStmt->lexNext();
                }
            }
            break;
        case FIND_NAME:
            if (searchStmt->variant() == FUNC_HEDR || searchStmt->variant() == PROC_HEDR)
            {
                if (!strcmp(searchStmt->symbol()->identifier(), funcName))
                    return searchStmt;
                else
                    searchStmt = searchStmt->lastNodeOfStmt()->lexNext();
            }
            else if (searchStmt->variant() == MODULE_PROC_STMT)
                searchStmt = searchStmt->lastNodeOfStmt()->lexNext();

            if (searchStmt->variant() == CONTROL_END) // end of interface block
            {
                mode = SEARCH_INTERFACE;
                searchStmt = searchStmt->lexNext();
            }
            break;
        }
    }
    return NULL;
}

SgStatement *getInterface(SgSymbol *s)
{
    SgStatement *func = cur_func;
    SgStatement *interface_st = NULL;
    while (func->variant() != GLOBAL)
    {
        if (interface_st = getInterfaceInScope(s, func))
            return interface_st;
        else
            func = func->controlParent();
    }
    return interface_st;
}

int CompareModuleProcedureName(SgExpression *name_list, SgSymbol *symb)
{
    SgExpression *el;
    for (el=name_list; el; el=el->rhs())
         if (!strcmp(el->lhs()->symbol()->identifier(), symb->identifier()))
             return 1;
    return 0;
} 

SgStatement *SearchModuleProcedure(SgExpression *name_list, SgExpression *arg_list, SgStatement *module_st)
{
    SgStatement *stmt = module_st->lexNext();
    while (stmt->variant() != CONTAINS_STMT && stmt->variant() != CONTROL_END )
        stmt = stmt->lastNodeOfStmt()->lexNext(); 
    if (stmt->variant() == CONTROL_END)
        return NULL;
    SgStatement *last = module_st->lastNodeOfStmt();
    for (stmt=stmt->lexNext(); stmt != last; stmt = stmt->lastNodeOfStmt()->lexNext())
    {   
        if (CompareModuleProcedureName(name_list, stmt->symbol()) && CompareArguments(stmt->symbol(),arg_list))
            return stmt;
        else
            continue;
    }
    return NULL;
}

SgStatement *getGenericInterfaceInScope(SgSymbol *s, SgExpression *arg_list, SgStatement *func)
{
    enum { SEARCH_INTERFACE, CHECK_INTERFACE, FIND_NAME };

    SgStatement *searchStmt = func->lexNext();
    SgStatement *tmp;
    const char *funcName = s->identifier();
    const char *toCmp;

    int mode = SEARCH_INTERFACE;
    //search interface in the specification part of a program unit 
    while (searchStmt && (!isSgExecutableStatement(searchStmt) || isDvmSpecification(searchStmt)))
    {
        switch (mode)
        {
        case SEARCH_INTERFACE:
            if (searchStmt->variant() != INTERFACE_STMT)
                searchStmt = searchStmt->lastNodeOfStmt()->lexNext();
            else
                mode = CHECK_INTERFACE;
            break;
        case CHECK_INTERFACE:
            if (searchStmt->symbol())
                toCmp = searchStmt->symbol()->identifier();
            else
                toCmp = "";

            if (searchStmt->symbol() && !strcmp(toCmp, funcName))
            {
                mode = FIND_NAME;
                searchStmt = searchStmt->lexNext();
            }
            else
            {
                searchStmt = searchStmt->lastNodeOfStmt()->lexNext();
                mode = SEARCH_INTERFACE;
            }
            break;
        case FIND_NAME:
            if (searchStmt->variant() == FUNC_HEDR || searchStmt->variant() == PROC_HEDR)
            {
                if (CompareArguments(searchStmt->symbol(), arg_list))
                    return searchStmt;
                else
                    searchStmt = searchStmt->lastNodeOfStmt()->lexNext();   
            }
            else if (searchStmt->variant() == MODULE_PROC_STMT)
            {    
                SgStatement *module_proc = SearchModuleProcedure(searchStmt->expr(0), arg_list, func->variant()==MODULE_STMT ? func : ORIGINAL_SYMBOL(searchStmt->expr(0)->symbol())->scope());
                if (module_proc)
                    return module_proc;
                else                 
                    searchStmt = searchStmt->lexNext();
            }
            if (searchStmt->variant() == CONTROL_END)  // end of interface block
            {   
                mode = SEARCH_INTERFACE;
                searchStmt = searchStmt->lexNext();
            }
            break;
        }
    }
    return NULL;
}

SgStatement *getGenericInterface(SgSymbol *s, SgExpression *arg_list)
{
    SgStatement *func = IS_BY_USE(s) ? ORIGINAL_SYMBOL(s)->scope() : cur_func;   
    SgStatement *interface_st = NULL;
    while (func->variant() != GLOBAL)
    {
        if (interface_st = getGenericInterfaceInScope(s, arg_list, func))
            return interface_st;
        else
            func = func->controlParent();
    }                                                   
    return interface_st;
}

void Call_Site(SgSymbol *s, int inlined, SgStatement *stat, SgExpression *e)
{
    graph_node * gnode, *node_by_attr = NULL;
    SgSymbol *s_new = s;
    //printf("\n%s  id= %d \n", s->identifier(), s->id());
    if (!do_dummy  && isDummyArgument(s)) return;
    if (!do_stmtfn && isStatementFunction(s)) return;
    // if(isIntrinsicFunction(s)) return; 
    //printf("\nLINE %d", cur_st->lineNumber());
   
    if(s->variant() == INTERFACE_NAME && in_region)
    {
        //printf("INTERFACE_NAME %s\n",s->identifier());
        SgStatement *interface_st = getGenericInterface(s, stat ? stat->expr(0) : e->lhs());
        SgSymbol *s_gen  = s;
        if(!interface_st)
        {
           Error("No interface found for the procedure %s", s->identifier(), 661, cur_func); 
           return;
        }
        s = interface_st->symbol();
        has_generic_interface = 1;
        if (stat)
            stat->setSymbol(*s);
        else
            e->setSymbol(*s); 
        MarkAsUserProcedure(s);
        MarkAsExternalProcedure(s);    
    }

    if (ATTR_NODE(s))
        node_by_attr = GRAPHNODE(s);
    gnode = GraphNode(s, NULL, 0);
    CreateOutcomingEdge(gnode, inlined); // for node 'cur_node' edge: [cur_node]-> gnode
    CreateIncomingEdge(gnode, inlined); // for node 'gnode'    edge:  cur_node ->[gnode]
    if(node_by_attr && gnode != node_by_attr)
    {
        s_new = &s->copy(); 
        if (stat)
            stat->setSymbol(*s_new);
        else
            e->setSymbol(*s_new); 
        graph_node **pnode = new (graph_node *);
        *pnode = gnode; 
        s_new->addAttribute(GRAPH_NODE, (void*)pnode, sizeof(graph_node *));
    }
    if (gnode->st_header)
        MarkAsUserProcedure(s_new);  
    //printf(" call site on line %d: %d %s: %d %d\n", stat ? stat->lineNumber() : 0, ATTR_NODE(s_new) ? GRAPHNODE(s_new)->id : -1, s_new->identifier(), s_new->id(), s->id());  
}

graph_node *GraphNode(SgSymbol *s, SgStatement *header_st, int flag_new)
{
    graph_node * gnode;
    graph_node **pnode = new (graph_node *);

#if __SPF
    addToCollection(__LINE__, __FILE__, pnode, 1);
#endif

    gnode = flag_new == NEW ? NULL : NodeForSymbInGraph(s, header_st);
    if (!gnode)
        gnode = NewGraphNode(s, header_st);

    *pnode = gnode;
    if (!ATTR_NODE(s)){
        s->addAttribute(GRAPH_NODE, (void*)pnode, sizeof(graph_node *));
        if (deb_reg > 1)
            printf("\n attribute NODE[%d] for %s[%d]\n", GRAPHNODE(s)->id, s->identifier(), s->id());
    }
    return(gnode);
}

graph_node *SearchOriginalSymbolNode(SgSymbol *s, graph_node *first_node)
{
    graph_node *ndl;
    SgSymbol * s_origin = ORIGINAL_SYMBOL(s);
    for (ndl = first_node; ndl->same_name_next; ndl = ndl->same_name_next) 
        if (ndl->file_id == current_file_id && ndl->symb->scope() == s_origin->scope())
            return (ndl);
    return (ndl);
}

graph_node *SearchInternalProcedureName(SgSymbol *s, SgStatement *proc_scope, graph_node *first_node)
{
    graph_node *ndl;
    for (ndl = first_node; ndl->same_name_next; ndl = ndl->same_name_next) 
    {
        if (ndl->type != 2) continue; // is not internal procedure
        if (ndl->file_id == current_file_id && ndl->symb->scope() == proc_scope)  
            return (ndl);
        else
            continue;
    }
    if (ndl->type == 2 && ndl->file_id == current_file_id && ndl->symb->scope() == proc_scope) 
        return (ndl);
    else
        return (NULL);

}

graph_node *SearchExternalProcedureName(graph_node *first_node)
{
    graph_node *ndl;
    for (ndl = first_node; ndl->same_name_next; ndl = ndl->same_name_next) 
        if (ndl->type == 1)
            return (ndl);
    if (ndl->type == 1)
        return (ndl);
    else
        return (NULL);  
}

graph_node *NodeForSymbInGraph(SgSymbol *s, SgStatement *stheader)
{
    graph_node *ndl, *node=NULL;
    for (ndl = node_list; ndl; ndl = ndl->next) {
        
        if (!strcmp(ndl->name, ORIGINAL_SYMBOL(s)->identifier()))
        {
            if(ndl->same_name_next)
            {
                if(IS_BY_USE(s))
                {
                    node = SearchOriginalSymbolNode(s, ndl);
                    return (node);
                }
                if( s->attributes() & EXTERNAL_BIT || getInterface(s))
                {
                    node = SearchExternalProcedureName(ndl); 
                    return (node); 
                }
                if (cur_func->controlParent()->variant() == GLOBAL)
                    node =  SearchInternalProcedureName(s, cur_func, ndl);
                else if (cur_func->controlParent()->variant() == MODULE_STMT)
                {
                    node =  SearchInternalProcedureName(s, cur_func, ndl);
                    if (!node)
                        node =  SearchInternalProcedureName(s, cur_func->controlParent(), ndl);
                }
                if (!node)
                    node = SearchExternalProcedureName(ndl); 
            } 
            else
                node = ndl;
          
            return(node);
        }
    }
    return(NULL);
}

graph_node *SameNameNode(char *name)
{
    graph_node *ndl;
    for (ndl = node_list->next; ndl; ndl = ndl->next)       
        if (!strcmp(ndl->name, name))
            return(ndl);
    return (NULL);
}

graph_node *NewGraphNode(SgSymbol *s, SgStatement *header_st)
{
    graph_node * gnode;

    gnode = new graph_node;
    gnode->id = ++gcount;
    gnode->next = node_list;
    node_list = gnode;
    gnode->same_name_next = SameNameNode(s->identifier());
    if (gnode->same_name_next)
        gnode->samenamed = gnode->same_name_next->samenamed = 1;        
    gnode->file = header_st ? current_file : NULL;
    gnode->file_id = header_st ? current_file_id : -1;
    gnode->st_header = header_st;
    gnode->symb = s;
    gnode->name = new char[strlen(s->identifier()) + 1];
#if __SPF
    addToCollection(__LINE__, __FILE__, gnode->name, 2);
#endif
    strcpy(gnode->name, s->identifier());
    gnode->to_called = NULL;
    gnode->from_calling = NULL;
    if (header_st && (header_st->variant() == FUNC_HEDR || header_st->variant() == PROC_HEDR))
    {
        if (header_st->controlParent()->variant() == MODULE_STMT)
            gnode->type = 3;
        else if (header_st->controlParent()->variant() == GLOBAL)
            gnode->type = 1;
        else
            gnode->type = 2;
    }
    else
        gnode->type = 0;
    if (header_st && header_st->expr(2))
    {
        if (header_st->expr(2)->variant() == PURE_OP)
            SYMB_ATTR(s->thesymb) = SYMB_ATTR(s->thesymb) | PURE_BIT;
        else if (header_st->expr(2)->variant() == ELEMENTAL_OP)
            SYMB_ATTR(s->thesymb) = SYMB_ATTR(s->thesymb) | ELEMENTAL_BIT;
    }
    gnode->split = 0;
    gnode->tmplt = 0;
    gnode->clone = 0;
    gnode->count = 0;
    gnode->is_routine = 0;
    gnode->st_interface = NULL;    
    //printf("%s --- %d %d\n",gnode->name,gnode->id,gnode->type);
    return(gnode);
}

edge *CreateOutcomingEdge(graph_node *gnode, int inlined)
{
    edge *out_edge, *edgl;
    //SgSymbol *sunit;
    //sunit = cur_func->symbol();

    // testing outcoming edge list of current (calling) routine graph-node: cur_node 
    for (edgl = cur_node->to_called; edgl; edgl = edgl->next)
    if ((edgl->to->symb == gnode->symb) && (edgl->inlined == inlined)) //there is outcoming edge: [cur_node]->gnode 
        return(edgl);
    // creating new edge: [cur_node]->gnode 
    out_edge = NewEdge(NULL, gnode, inlined);   //NULL -> cur_node
    out_edge->next = cur_node->to_called;
    cur_node->to_called = out_edge;
    return(out_edge);
}

edge *CreateIncomingEdge(graph_node *gnode, int inlined)
{
    edge *in_edge, *edgl;
    //SgSymbol *sunit;
    //sunit = cur_func->symbol();

    // testing incoming edge list of called routine graph-node: gnode 
    for (edgl = gnode->from_calling; edgl; edgl = edgl->next)
    if ((edgl->from->symb == cur_node->symb) && (edgl->inlined == inlined)) //there is incoming edge: : cur_node->[gnode] 
        return(edgl);
    // creating new edge: cur_node->[gnode]
    in_edge = NewEdge(cur_node, NULL, inlined);   //NULL -> gnode
    in_edge->next = gnode->from_calling;
    gnode->from_calling = in_edge;
    return(in_edge);
}

edge *NewEdge(graph_node *from, graph_node *to, int inlined)
{
    edge *nedg;
    nedg = new edge;
    nedg->from = from;
    nedg->to = to;
    nedg->inlined = inlined;
    return(nedg);
}

/**********************************************************************/

/*    Testing and Help Functions                                      */

/**********************************************************************/


int isDummyArgument(SgSymbol *s)
{
    if (s->thesymb->entry.var_decl.local == IO)  // is dummy argument
        return(1);
    else
        return(0);
}

int isHeaderStmtSymbol(SgSymbol *s)
{
    return(DECL(s) == 1 && (s->variant() == FUNCTION_NAME || s->variant() == PROCEDURE_NAME || s->variant() == PROGRAM_NAME));
}

int isStatementFunction(SgSymbol *s)
{
    if (DECL(s) == 2)
        //if(s->scope() == cur_func && s->variant()==FUNCTION_NAME) 
        return (1); //is statement function symbol
    else return (0);
}

int isHeaderNode(graph_node *gnode)
{
    //header node represent a "top level" routine:
    //main program, or any subprogram which was called 
    //without inline expansion somewhere in the original program 
    edge * edgl;
    if (gnode->symb->variant() == PROGRAM_NAME)
        return(1);
    for (edgl = gnode->from_calling; edgl; edgl = edgl->next)
    if (!edgl->inlined) return(1);
    return(0);
}

int isDeadNode(graph_node *gnode)
{
    // dead node represent a "dead" routine:
    // a subprogram which was not called
    if (gnode->from_calling || gnode->symb->variant() == PROGRAM_NAME)
        return(0);
    else
        return(1);
}

int isNoBodyNode(graph_node *gnode)
{
    // nobody node represent a "nobody" routine: intrinsic or absent

    if (gnode->st_header)
        return(0);
    else
        return(1);
}


graph_node_list  *addToNodeList(graph_node_list *pnode, graph_node *gnode)
{
    // adding the node to the beginning of node list 
    // pnode-> gnode -> gnode-> ... -> gnode     
    graph_node_list * ndl;
    if (!pnode) {
        pnode = new graph_node_list;
        pnode->node = gnode;
        pnode->next = NULL;
    }
    else {
        ndl = new graph_node_list;
        ndl->node = gnode;
        ndl->next = pnode;
        pnode = ndl;
    }
    return (pnode);
}

graph_node_list  *delFromNodeList(graph_node_list *pnode, graph_node *gnode)
{
    // deleting the node from the node list 

    graph_node_list * ndl, *l;
    if (!pnode) return (NULL);
    if (pnode->node == gnode) return(pnode->next);
    l = pnode;
    for (ndl = pnode->next; ndl; ndl = ndl->next)
    {
        if (ndl->node == gnode)
        {
            l->next = ndl->next;
            return(pnode);
        }
        else
            l = ndl;
    }
    return (pnode);
}

graph_node_list  *isInNodeList(graph_node_list *pnode, graph_node *gnode)
{
    // testing: is there node in the node list 

    graph_node_list * ndl;
    if (!pnode) return (NULL);
    for (ndl = pnode; ndl; ndl = ndl->next)
    {
        if (ndl->node == gnode)
            return(ndl);
    }
    return (NULL);
}


void PrintGraphNode(graph_node *gnode)
{
    edge * edgl;
    printf("\n%s(%d)[%d]  ->   ", gnode->name, gnode->symb->id(), gnode->id);
    for (edgl = gnode->to_called; edgl; edgl = edgl->next)
        printf("   %s(%d)", edgl->to->name, edgl->to->symb->id());
}

void PrintGraphNodeWithAllEdges(graph_node *gnode)
{
    edge * edgl;
    printf("\n");
    for (edgl = gnode->from_calling; edgl; edgl = edgl->next)
        printf("   %s(%d)", edgl->from->name, edgl->from->symb->id());
    if (!gnode->from_calling)
        printf("          ");
    printf("   ->%s(%d)->   ", gnode->name, gnode->symb->id());
    for (edgl = gnode->to_called; edgl; edgl = edgl->next)
        printf("   %s(%d)", edgl->to->name, edgl->to->symb->id());
}

void PrintWholeGraph()
{
    graph_node *ndl;
    printf("\n%s\n", "C a l l  G r a p h");
    for (ndl = node_list; ndl; ndl = ndl->next)
        PrintGraphNode(ndl);
    printf("\n");
}

void PrintWholeGraph_kind_2()
{
    graph_node *ndl;
    printf("\n%s\n", "C a l l  G r a p h  2");
    for (ndl = node_list; ndl; ndl = ndl->next)
        PrintGraphNodeWithAllEdges(ndl);
    printf("\n");
}


void DeleteIncomingEdgeFrom(graph_node *gnode, graph_node *from)
{
    // deleting edge that is incoming to node 'gnode' from node 'from' 
    edge *edgl, *ledge;
    ledge = NULL;
    for (edgl = gnode->from_calling; edgl; edgl = edgl->next) {
        if (edgl->from == from) {
            if (deb_reg > 1)
                printf("\n%s(%d)-%s(%d) edge dead  ", from->name, from->symb->id(), gnode->name, gnode->symb->id());

            if (ledge)
                ledge->next = edgl->next;
            else
                gnode->from_calling = edgl->next;
        }
        else
            ledge = edgl;
    }
}

void DeleteOutcomingEdgeTo(graph_node *gnode, graph_node *gto)
{
    // deleting edge that is outcoming from node 'gnode' to node 'gto' 
    edge *edgl, *ledge;
    ledge = NULL;
    for (edgl = gnode->to_called; edgl; edgl = edgl->next) {
        if (edgl->to == gto) {
            if (deb_reg > 1)
                printf("\n%s(%d)-%s(%d) edge empty  ", gnode->name, gnode->symb->id(), gto->name, gto->symb->id());

            if (ledge)
                ledge->next = edgl->next;
            else
                gnode->to_called = edgl->next;
        }
        else
            ledge = edgl;
    }
}

void ScanSymbolTable(SgFile *f)
{
    SgSymbol *s;
    for (s = f->firstSymbol(); s; s = s->next())
        //if(isHeaderStmtSymbol(s))
        printSymb(s);
}

void ScanTypeTable(SgFile *f)
{
    SgType *t;
    for (t = f->firstType(); t; t = t->next())
    {   // printf("TYPE[%d] : ", t->id());
        printType(t);
    }
}

void ReseatEdges(graph_node *gnode, graph_node *newnode)
{//reseat all edges representing inlined calls to gnode to point to newnode 
    edge *edgl, *tol, *ledge, *curedg;
    graph_node *from;
    ledge = NULL;
    // for(edgl=gnode->from_calling; edgl; edgl=edgl->next)
    // looking through the incoming edge list of gnode
    edgl = gnode->from_calling;
    while (edgl)
    {
        if (edgl->inlined)
        {
            from = edgl->from;
            // reseating outcoming edge to 'gnode' to point to 'newnode'
            for (tol = from->to_called; tol; tol = tol->next)
            if (tol->to == gnode && tol->inlined)
            {
                tol->to = newnode; break;
            }
            // removing "inlined" incoming edge of gnode
            if (ledge)
                ledge->next = edgl->next;
            else
                gnode->from_calling = edgl->next;

            curedg = edgl;    // set curedg to point at removed edge  
            edgl = edgl->next;  // to next node of list

            // adding removed edge  to 'newnode' 
            curedg->next = newnode->from_calling;
            newnode->from_calling = curedg;

        }
        else
        {
            ledge = edgl;
            edgl = edgl->next;
        }
    } //end while  
}

void CopyOutcomingEdges(graph_node *gnode, graph_node *gnew)
{
    edge *out_edge, *in_edge, *edgl;
    graph_node *s;
    // looking through the outcoming edge list of gnode
    for (edgl = gnode->to_called; edgl; edgl = edgl->next)
    {
        s = edgl->to; // successor of gnode
        // creating new edge of gnew (copy of edgl)
        out_edge = NewEdge(NULL, edgl->to, edgl->inlined);
        out_edge->next = gnew->to_called;
        gnew->to_called = out_edge;
        // creating new edge of  s (successor  of gnode)
        in_edge = NewEdge(gnew, NULL, edgl->inlined);
        in_edge->next = s->from_calling;
        s->from_calling = in_edge;
    }
    return;
}

void CopyIncomingEdges(graph_node *gnode, graph_node *gnew)
{
    edge *in_edge, *out_edge, *edgl;
    graph_node *p;
    // looking through the incoming edge list of gnode
    for (edgl = gnode->from_calling; edgl; edgl = edgl->next)
    {
        p = edgl->from; // predecessor of gnode
        // creating new edge of gnew (copy of edgl) 
        in_edge = NewEdge(edgl->from, NULL, edgl->inlined);
        in_edge->next = gnew->from_calling;
        gnew->from_calling = in_edge;
        // creating new edge of p (predecessor of gnode)
        out_edge = NewEdge(NULL, gnew, edgl->inlined);
        out_edge->next = p->to_called;
        p->to_called = out_edge;

    }
    return;
}

void printSymb(SgSymbol *s)
{
    const char *head;
    head = isHeaderStmtSymbol(s) ? "HEADER  " : "        ";
    printf("SYMB[%3d]  scope=STMT[%3d] : %s    %s", s->id(), (s->scope()) ? (s->scope())->id() : -1, s->identifier(), head);
    printType(s->type());
    if(IS_BY_USE(s))
       printf(" BY_USE %s", ORIGINAL_SYMBOL(s)->scope()->symbol()->identifier());
    if(ATTR_NODE(s))
       printf(" GRAPHNODE %d", GRAPHNODE(s)->id);
    printf("\n");
}

void printType(SgType *t)
{
    SgArrayType *arrayt;

    if (!t) {
        printf("no type "); return;
    }
    else   printf("TYPE[%d]:", t->id());
    if ((arrayt = isSgArrayType(t)) != 0)
    {
        SgExpression *e = arrayt->getDimList();
        if (!e)
            printf(" dimension() ");
        else
            printf(" dimension(%s) ", UnparseExpr(arrayt->getDimList()));
        /*
          int i;
          int n = arrayt->dimension();
          printf("dimension(");
          for(i = 0; i < n; i++)
          { if(arrayt->sizeInDim(i))
          { printf("%s", UnparseExpr(arrayt->sizeInDim(i))); //(arrayt->sizeInDim(i))->unparsestdout();
          if(i < n-1)  printf(", ");
          }
          }
          printf(") ");
          */
    }
    else
    {
        switch (t->variant())
        {
        case  T_INT:      printf("integer "); break;
        case  T_FLOAT:    printf("real "); break;
        case  T_DOUBLE:   printf("double precision "); break;
        case  T_CHAR:     printf("character "); break;
        case  T_STRING:   printf("Character ");
            UnparseLLND(TYPE_RANGES(t->thetype));
            /*if(t->length()) printf("[%d]",t->length()->variant());*/
            /*((SgArrayType *) t)->getDimList()->unparsestdout();*/
            break;
        case  T_BOOL:     printf("logical "); break;
        case  T_COMPLEX:  printf("complex "); break;
        case  T_DCOMPLEX:  printf("double complex "); break;

        default: break;
        }
    }

    if (t->hasBaseType())
    {
        printf("of ");
        printType(t->baseType());
    }
}

#undef NEW