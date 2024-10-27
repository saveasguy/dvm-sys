
#include <stdio.h>

#include <stdlib.h>
#include <stdarg.h> /* ANSI variable argument header */

#include "compatible.h"   /* Make different system compatible... (PHB) */
#ifdef SYS5
#include <string.h>
#else
#include <strings.h>
#endif

#include "vpc.h"
#include "macro.h"
#include "ext_lib.h"
#define HPF_STRING 2

/*int debug =NO;  used in db.c*/

/*static int number_of_bif_node = 0;*/
/*int number_of_ll_node = 0; this counters are useless anymore */
/*static int number_of_symb_node  = 0;
static int number_of_type_node = 0;*/

/* FORWARD DECLARATIONS (phb) */
void Message();
char * filter();
int *evaluateExpression();

extern int write_nodes();
extern char* Tool_Unparse2_LLnode(); 
extern void Init_HPFUnparser();
extern char* Tool_Unparse_Bif ();
extern char* Tool_Unparse_Type();

#define MAXFIELDSYMB 10
#define MAXFIELDTYPE 10

#ifdef __SPF_BUILT_IN_PARSER
static int Warning_count = 0;
//static PTR_FILE pointer_on_file_proj;
//static char* default_filename;

/* records propoerties and type of node */
//static char node_code_type[LAST_CODE];
/* Number of argument-words in each kind of tree-node.  */
//static int node_code_length[LAST_CODE];
//static enum typenode node_code_kind[LAST_CODE];

/* special table for infos on type and symbol */
static char info_type[LAST_CODE][MAXFIELDTYPE];
static char info_symb[LAST_CODE][MAXFIELDSYMB];
static char general_info[LAST_CODE][MAXFIELDSYMB];
#else
PTR_FILE pointer_on_file_proj;
char* default_filename;
int Warning_count = 0;

/* records propoerties and type of node */
char node_code_type[LAST_CODE];
/* Number of argument-words in each kind of tree-node.  */
int node_code_length[LAST_CODE];
enum typenode node_code_kind[LAST_CODE];

/* special table for infos on type and symbol */
char info_type[LAST_CODE][MAXFIELDTYPE];
char info_symb[LAST_CODE][MAXFIELDSYMB];
char general_info[LAST_CODE][MAXFIELDSYMB];
#endif

/*static struct bif_stack_level   *stack_level = NULL;
static struct bif_stack_level *current_level = NULL;*/

PTR_BFND  getFunctionHeader();

/*****************************************************************************
 *                                                                           *
 *                   Procedure of general use                                *
 *                                                                           *
 *****************************************************************************/

/* Modified to return a pointer (64bit clean) (phb) */
/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
char* xmalloc_temp(size)
#else
char* xmalloc(size)
#endif
     int size;
{
  char *val;
  val = (char *) malloc (size);
  
  if (val == 0)
    Message("Virtual memory exhausted (malloc failed)",0);
  return val;
}

/* list of allocated data */
static ptstack_chaining Current_Allocated_Data = NULL;
static ptstack_chaining First_STACK= NULL;

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
void make_a_malloc_stack_temp()
#else
void make_a_malloc_stack()
#endif
{
  ptstack_chaining pt;
    
  pt = (ptstack_chaining) malloc(sizeof(struct stack_chaining));
  if (!pt)
    {
      Message("sorry : out of memory\n",0);
      exit(1);
    }
  
  if (Current_Allocated_Data)
    Current_Allocated_Data->next = pt;
  pt->first = NULL;
  pt->last = NULL;
  pt->prev = Current_Allocated_Data;
  if (Current_Allocated_Data)
    pt->level = Current_Allocated_Data->level +1;
  else
    pt->level = 0;
/*  printf("make_a_malloc_stack %d \n",pt->level);*/
  Current_Allocated_Data = pt;
  if (First_STACK == NULL)
    First_STACK = pt;
}

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
char* mymalloc_temp(size)
#else
char* mymalloc(size)
#endif
int size;
{
  char *pt1;
  ptchaining pt2;
  if (!Current_Allocated_Data)
    {
      Message("Allocated Stack not defined\n",0);
      exit(1);
    }

/*  if (Current_Allocated_Data->level > 0)
    printf("mymalloc  %d \n", Current_Allocated_Data->level); */
  pt1 = (char *) malloc(size);
  if (!pt1)
    {
      Message("sorry : out of memory\n",0);
      exit(1);
    }
 
  pt2 = (ptchaining) malloc(sizeof(struct chaining));
  if (!pt2 )
    {
      Message("sorry : out of memory\n",0);
      exit(1);
    }
  
  pt2->zone = pt1;
  pt2->list = NULL;

  if (Current_Allocated_Data->first == NULL)
    Current_Allocated_Data->first = pt2;
  
  if (Current_Allocated_Data->last == NULL)
    Current_Allocated_Data->last = pt2;
  else
    {
      Current_Allocated_Data->last->list = pt2;
      Current_Allocated_Data->last = pt2;
    }
  return pt1;
}

/***************** Provides infos on nodes ********************************
 *                                                                        *
 *     based on the table info in include dir *.def                       *
 *                                                                        *
 **************************************************************************/

#ifndef __SPF_BUILT_IN_PARSER
/***************************************************************************/
int isATypeNode(variant)
int variant;
{
  return (TYPENODE == (int) node_code_kind[variant]);
}

/***************************************************************************/
int isASymbNode(variant)
int variant;
{
  return (SYMBNODE == (int) node_code_kind[variant]);
}

/***************************************************************************/
int isABifNode(variant)
int variant;
{
  return (BIFNODE == (int) node_code_kind[variant]);
}

/***************************************************************************/
int isALoNode(variant)
int variant;
{
  return (LLNODE == (int) node_code_kind[variant]);
}
#endif

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int hasTypeBaseType_temp(variant)
#else
int hasTypeBaseType(variant)
#endif
int variant;
{
  if (!isATypeNode(variant))
    {
      Message("hasTypeBaseType not applied to a type node",0);
      return 0;
    }
  if (info_type[variant][2] == 'b')
    return TRUE;
  else
    return FALSE;
}

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int isStructType_temp(variant)
#else
int isStructType(variant)
#endif
int variant;
{
  if (!isATypeNode(variant))
    {
      Message("isStructType not applied to a type node",0);
      return 0;
    }
  if (info_type[variant][0] == 's')
    return TRUE;
  else
    return FALSE;
}

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int isPointerType_temp(variant)
#else
int isPointerType(variant)
#endif
int variant;
{
  if (!isATypeNode(variant))
    {
      Message("isPointerType not applied to a type node",0);
      return 0;
    }
  if (info_type[variant][0] == 'p')
    return TRUE;
  else
    return FALSE;
}


/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int isUnionType_temp(variant)
#else
int isUnionType(variant)
#endif
int variant;
{
  if (!isATypeNode(variant))
    {
      Message("isUnionType not applied to a type node",0);
      return 0;
    }
  if (info_type[variant][0] == 'u')
    return TRUE;
  else
    return FALSE;
}


/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int isEnumType_temp(variant)
#else
int isEnumType(variant)
#endif
int variant;
{
  if (!isATypeNode(variant))
    {
      Message("EnumType not applied to a type node",0);
      return 0;
    }
  if (info_type[variant][0] == 'e')
    return TRUE;
  else
    return FALSE;
}


/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int hasTypeSymbol_temp(variant)
#else
int hasTypeSymbol(variant)
#endif
int variant;
{
  if (!isATypeNode(variant))
    {
      Message("hasTypeSymbol not applied to a type node",0);
      return 0;
    }
  if (info_type[variant][1] == 's')
    return TRUE;
  else
    return FALSE;
}

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int isAtomicType_temp(variant)
#else
int isAtomicType(variant)
#endif
int variant;
{
  if (!isATypeNode(variant))
    {
      Message("isAtomicType not applied to a type node",0);
      return 0;
    }
  if (info_type[variant][0] == 'a')
    return TRUE;
  else
    return FALSE;
}

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int hasNodeASymb_temp(variant)
#else
int hasNodeASymb(variant)
#endif
int variant;
{
  if ((!isABifNode(variant)) && (!isALoNode(variant)))
    {
      Message("hasNodeASymb not applied to a bif or low level node",0);
      return 0;
    }
  if (general_info[variant][2] == 's')
    return TRUE;
  else
    return FALSE;
}

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int isNodeAConst_temp(variant)
#else
int isNodeAConst(variant)
#endif
int variant;
{
  if ((!isABifNode(variant)) && (!isALoNode(variant)))
    {
      Message("isNodeAConst not applied to a bif or low level node",0);
      return 0;
    }
  if (general_info[variant][1] == 'c')
    return TRUE;
  else
    return FALSE;
}


/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int isAStructDeclBif_temp(variant)
#else
int isAStructDeclBif(variant)
#endif
int variant;
{
  if (!isABifNode(variant))
    {
      Message("isAStructDeclBif not applied to a bif",0);
      return 0;
    }
  if (general_info[variant][1] == 's')
    return TRUE;
  else
    return FALSE;
}

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int isAUnionDeclBif_temp(variant)
#else
int isAUnionDeclBif(variant)
#endif
int variant;
{
  if (!isABifNode(variant))
    {
      Message("isAUnionDeclBif not applied to a bif",0);
      return 0;
    }
  if (general_info[variant][1] == 'u')
    return TRUE;
  else
    return FALSE;
}

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int isAEnumDeclBif_temp(variant)
#else
int isAEnumDeclBif(variant)
#endif
int variant;
{
  if (!isABifNode(variant))
    {
      Message("isAEnumDeclBif not applied to a bif",0);
      return 0;
    }
  if (general_info[variant][1] == 'e')
    return TRUE;
  else
    return FALSE;
}

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int isADeclBif_temp(variant)
#else
int isADeclBif(variant)
#endif
int variant;
{
  if (!isABifNode(variant))
    {
      Message("isADeclBif not applied to a bif",0);
      return 0;
    }
  if (general_info[variant][0] == 'd')
    return TRUE;
  else
    return FALSE;
}

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int isAControlEnd_temp(variant)
#else
int isAControlEnd(variant)
#endif
int variant;
{
  if (!isABifNode(variant))
    {
      Message("isAControlEnd not applied to a bif",0);
      return 0;
    }
  if (general_info[variant][0] == 'c')
    return TRUE;
  else
    return FALSE;
}


/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
void Message_temp(s, l)
#else
void Message(s, l)
#endif
char *s;
int l;
{
  if (l != 0)
    fprintf(stderr,"Warning : %s line %d\n",s, l);
  else
    fprintf(stderr,"Warning : %s\n",s);
 Warning_count++;
}
/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int Check_Lang_Fortran_temp(proj)
#else
int Check_Lang_Fortran(proj)
#endif
PTR_PROJ proj;
{
  PTR_FILE ptf;
  PTR_BLOB ptb;
  /* Change FALSE to TRUE */
  if (!proj)
    return TRUE;
  for (ptb = PROJ_FILE_CHAIN (proj); ptb ; ptb =  BLOB_NEXT (ptb))
    {
      ptf = (PTR_FILE) BLOB_VALUE (ptb);
    /*  if (debug)
	fprintf(stderr,"%s\n",FILE_FILENAME (ptf)); */

      if (FILE_LANGUAGE(ptf) != ForSrc)
	return(FALSE);
    }
  return(TRUE);
}


/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
char* filter_temp(s)
#else
char* filter(s)
#endif
char *s;
{
  char c;
  int i = 1;
  char temp[1024];
  int temp_i = 0;
  int buf_i = 0;
  int commentline = 0;
  char *resul, *init;
  
  if (!s) return NULL;
  if (strlen(s)==0) return s;

  /* allocate very simple, but to redo later, allocate two times the size */
  make_a_malloc_stack();
  resul = (char *) mymalloc((int)(2*strlen(s)));
  memset(resul, 0, 2*strlen(s));
  init = resul;
  /* find the separator */
  c = s[0];

  if ((c != ' ')
      &&(c != '\n')
      && (c != '0')
      && (c != '1') 
      && (c != '2') 
      && (c != '3') 
      && (c != '4') 
      && (c != '5') 
      && (c != '6') 
      && (c != '7') 
      && (c != '8') 
      && (c != '9'))
      if ( c=='!')  commentline = HPF_STRING;
      else commentline = 1;
  else
    commentline = 0;

  temp_i = 0;
  i = 0;
  buf_i =0;
  while (c!='\0')
    {
      c = s[i];
      temp[ buf_i] = c;
      if (c == '\n')
        {
          temp[ buf_i+1] = '\0';
          sprintf(resul,"%s",temp);
          resul = resul + strlen(temp);
          temp_i = -1;
          buf_i = -1;
          if ((s[i+1] != ' ')
              &&(s[i+1] != '\n')
              && (s[i+1] != '0')
              && (s[i+1] != '1') 
              && (s[i+1] != '2') 
              && (s[i+1] != '3') 
              && (s[i+1] != '4') 
              && (s[i+1] != '5') 
              && (s[i+1] != '6') 
              && (s[i+1] != '7') 
              && (s[i+1] != '8') 
              && (s[i+1] != '9'))
	      if (s[i+1] == '!')  commentline = HPF_STRING;
	      else commentline = 1;
          else
            commentline = 0;
        } else
          {
            if ((temp_i == 71) && !commentline)
              { 
                /* insert where necessary */
                temp[ buf_i+1]  = '\0';
                sprintf(resul,"%s\n",temp);
                resul = resul + strlen(temp)+1;
                sprintf(resul,"     +");
                resul = resul + strlen("     +");
                commentline = 0;
                memset(temp, 0, 1024);
                temp_i = strlen("     +")-1;
                buf_i = -1;
              }
	    if ((temp_i == 71) && (commentline==HPF_STRING))
              { 
                /* insert where necessary */
		int count=0;
		for(;s[i]!='$';i--,count++)
            	    {
		    if (strncmp(&(s[i]),"ONTO", strlen("ONTO"))== 0)
			break;
		    if (strncmp(&(s[i]),"BEGIN", strlen("BEGIN"))== 0)
			break;
		    if (strncmp(&(s[i]),"ON", strlen("ON"))== 0)
			{
			i+=3;count-=3;		
			break;
			}
		    if (strncmp(&(s[i]),"WITH", strlen("WITH"))== 0)
			break;
		    if (strncmp(&(s[i]),"NEW", strlen("NEW"))== 0)
			break;
		    if (strncmp(&(s[i]),"REDUCTION", strlen("REDUCTION"))== 0)
			break;
		    if (strncmp(&(s[i]),"TEMPLATE", strlen("TEMPLATE"))== 0)
			break;
		    if (strncmp(&(s[i]),"SHADOW", strlen("SHADOW"))== 0)
			break;
		    if (strncmp(&(s[i]),"INHERIT", strlen("INHERIT"))== 0)
			break;
		    if (strncmp(&(s[i]),"DYNAMIC", strlen("DYNAMIC"))== 0)
			break;
		    if (strncmp(&(s[i]),"DIMENSION", strlen("DIMENSION"))== 0)
			break;
		    if (strncmp(&(s[i]),"PROCESSORS", strlen("PROCESSORS"))== 0)
			break;
		    if (strncmp(&(s[i]),"DISTRIBUTE", strlen("DISTRIBUTE"))== 0)
			break;
		    if (strncmp(&(s[i]),"ALIGN", strlen("ALIGN"))== 0)
			break;
		    if (strncmp(&(s[i]),"::", strlen("::"))== 0)
			{
			/*i+=3;count-=3;*/
			break;
			}
		    }
		i--;count++;      
		if (count<36)
		    temp[ buf_i+1-count]  = '\0';
		else 
		    {
		    i+=count;
		    temp[ buf_i+1]  = '\0';
		    }
                sprintf(resul,"%s\n",temp);
                resul = resul + strlen(temp)+1;
                sprintf(resul,"!HPF$*");
                resul = resul + strlen("!HPF$*");
                /*8commentline = 0;*/
                memset(temp, 0, 1024);
                temp_i = strlen("!HPF$*")-1;
                buf_i = -1;
              }
          }
      i++;
      temp_i++;
      buf_i++;
    }

  return init;  
}
/* preset some values of symbols for evaluateExpression*/
#define ALLOCATECHUNKVALUE  100
static PTR_SYMB  *ValuesSymb = NULL;
static int       *ValuesInt  = NULL;
static int        NbValues   = 0;
static int        NbElement   = 0;

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
void allocateValueEvaluate_temp()
#else
void allocateValueEvaluate()
#endif
{
  int i;
  PTR_SYMB  *pt1;
  int       *pt2;
  
  pt1 = (PTR_SYMB  *) xmalloc((int)( sizeof(PTR_SYMB  *) * 
			      (NbValues + ALLOCATECHUNKVALUE)));
  pt2 =  (int *) xmalloc((int)( sizeof(int  *) * (NbValues + ALLOCATECHUNKVALUE)));
  
  for (i=0; i<NbValues + ALLOCATECHUNKVALUE; i++) {
    pt1[i] = NULL;
    pt2[i] = 0;
  }
  
  for (i=0 ; i < NbValues; i++) {
    pt1[i] = ValuesSymb[i];
    pt2[i] = ValuesInt[i];
  }
  
  if (NbValues) {
#ifdef __SPF
      removeFromCollection(ValuesSymb);
      removeFromCollection(ValuesInt);
#endif
    free(ValuesSymb);
    free(ValuesInt);
  }
    
  ValuesSymb = pt1;
  ValuesInt = pt2;
  NbValues = NbValues + ALLOCATECHUNKVALUE;
}

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
void addElementEvaluate_temp(symb, val)
#else
void addElementEvaluate(symb, val)
#endif
     PTR_SYMB symb;
     int val;
{
  if (!symb)
    return;
  while (NbValues <= ( NbElement+1))
    {
      allocateValueEvaluate();
    }
  ValuesSymb[NbElement] = symb;
  ValuesInt[NbElement] = val;
  NbElement++;
}


/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int getElementEvaluate_temp(symb)
#else
int getElementEvaluate(symb)
#endif
     PTR_SYMB symb;
{
  int i;
  if (!symb)
    return -1;
  for (i=0 ; i < NbElement; i++)
    {
      if (ValuesSymb[i] == symb)
        return i;
    }
  return -1;
}


/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
void resetPresetEvaluate_temp()
#else
void resetPresetEvaluate()
#endif
{
  NbValues = 0;
  NbElement  = 0;
  if (ValuesSymb)
  {
#ifdef __SPF
      removeFromCollection(ValuesSymb);
#endif
      free(ValuesSymb);
  }
  if (ValuesInt)
  {
#ifdef __SPF
      removeFromCollection(ValuesInt);
#endif
      free(ValuesInt);
  }
  ValuesSymb = NULL;
  ValuesInt = NULL;
}

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int* evaluateExpression_temp(expr)
#else
int* evaluateExpression(expr)
#endif
     PTR_LLND expr;
{
  int *res, *op1, *op2;
  
  res = (int *) xmalloc((int)(2 * sizeof(int)));
  memset((char *) res, 0, 2 * sizeof (int));
  op1 = (int *) xmalloc((int)(2 * sizeof(int)));
  memset((char *)op1, 0, 2 * sizeof (int));
  op2 = (int *) xmalloc((int)(2 * sizeof(int)));
  memset((char *) op2, 0, 2 * sizeof (int));
  if (! expr)
    {
      res [0] = -1;
      return res;
    }
  
  switch (BIF_CODE (expr)) 
    {
    case INT_VAL:
      res [1] = NODE_INT_CST_LOW (expr);
#ifdef __SPF
      removeFromCollection(op1);
      removeFromCollection(op2);
#endif
      free(op1); free(op2);
      return res;
    case ADD_OP :
      op1  = evaluateExpression (NODE_OPERAND0 (expr));
      op2  = evaluateExpression (NODE_OPERAND1 (expr));
      if ((op1 [0] == -1) || (op2 [0] == -1))
        res [0] = -1;
      else
        res [1] = op1 [1] + op2 [1];
#ifdef __SPF
      removeFromCollection(op1);
      removeFromCollection(op2);
#endif
      free(op1); free(op2);
      return res;
   case MULT_OP :
      op1  = evaluateExpression (NODE_OPERAND0 (expr));
      op2  = evaluateExpression (NODE_OPERAND1 (expr));
      if ((op1 [0] == -1) || (op2 [0] == -1))
        res [0] = -1;
      else
        res [1] = op1 [1] * op2 [1];
#ifdef __SPF
      removeFromCollection(op1);
      removeFromCollection(op2);
#endif
      free(op1); free(op2);
      return res;
   case SUBT_OP :
      op1  = evaluateExpression (NODE_OPERAND0 (expr));
      op2  = evaluateExpression (NODE_OPERAND1 (expr));
      if ((op1 [0] == -1) || (op2 [0] == -1))
        res [0] = -1;
      else
        res [1] = op1 [1] - op2 [1];
#ifdef __SPF
      removeFromCollection(op1);
      removeFromCollection(op2);
#endif
      free(op1); free(op2);
      return res;
   case DIV_OP :
        op1  = evaluateExpression (NODE_OPERAND0 (expr));
      op2  = evaluateExpression (NODE_OPERAND1 (expr));
      if ((op1 [0] == -1) || (op2 [0] == -1))
        res [0] = -1;
      else
        res [1] = op1 [1] / op2 [1];
#ifdef __SPF
      removeFromCollection(op1);
      removeFromCollection(op2);
#endif
      free(op1); free(op2);
      return res;
   case MOD_OP :
      op1  = evaluateExpression (NODE_OPERAND0 (expr));
      op2  = evaluateExpression (NODE_OPERAND1 (expr));
      if ((op1 [0] == -1) || (op2 [0] == -1))
        res [0] = -1;
      else
        res [1] = op1 [1] % op2 [1];
#ifdef __SPF
      removeFromCollection(op1);
      removeFromCollection(op2);
#endif
      free(op1); free(op2);
      return res;
   case EXP_OP :
      op1  = evaluateExpression (NODE_OPERAND0 (expr));
      op2  = evaluateExpression (NODE_OPERAND1 (expr));
      if ((op1 [0] == -1) || (op2 [0] == -1))
        res [0] = -1;
      else
        res [1] = op1 [1] ^ op2 [1];
#ifdef __SPF
      removeFromCollection(op1);
      removeFromCollection(op2);
#endif
      free(op1); free(op2);
      return res;

    case MINUS_OP :
      op1  = evaluateExpression (NODE_OPERAND0 (expr));
      if (op1 [0] == -1)
        res [0] = -1;
      else
        res [1] = - op1 [1];  
#ifdef __SPF
      removeFromCollection(op1);
#endif
      free(op1);
      return res;
    case VAR_REF: /* assume here that some value for Symbole are given*/
      {
        int ind;
        if ((ind = getElementEvaluate(NODE_SYMB(expr))) != -1)
          {
             res [1] = ValuesInt[ind];
#ifdef __SPF
             removeFromCollection(op1);
             removeFromCollection(op2);
#endif
             free(op1); free(op2);
             return res;
           } else
             {
                res [0] = -1;
#ifdef __SPF
                removeFromCollection(op1);
                removeFromCollection(op2);
#endif
                free(op1); free(op2);
                return res;
             }
        break;
      }
   default :
     res [0] = -1;
#ifdef __SPF
     removeFromCollection(op1);
     removeFromCollection(op2);
#endif
      free(op1); free(op2);
      return res;
    }
    return res;
}

/***************************************************************************/
#ifdef __SPF_BUILT_IN_PARSER
int patternMatchExpression_temp(ll1, ll2)
#else
int patternMatchExpression(ll1,ll2)
#endif
     PTR_LLND ll1,ll2;
{
  int *res1, *res2;
  
  if (ll1 == ll2)
    return TRUE;
  
 if (!ll1 || !ll2)
   return FALSE;

  if (NODE_CODE(ll1) != NODE_CODE(ll2))
    return FALSE;
  /* first test if constant equations identical */
  res1 = evaluateExpression(ll1);
  res2 = evaluateExpression(ll2);
  if ((res1[0] != -1) &&
      (res2[0] != -1) &&
      (res1[1] == res2[1]))
    {
#ifdef __SPF
      removeFromCollection(res1);
      removeFromCollection(res2);
#endif
      free(res1);
      free(res2);
      return TRUE;
    }
  if ((res1[0] != -1) && (res2[0] == -1))
    {
#ifdef __SPF
      removeFromCollection(res1);
      removeFromCollection(res2);
#endif
      free(res1);
      free(res2);
      return FALSE;
    }
  if ((res1[0] == -1) && (res2[0] != -1))
    {
#ifdef __SPF
      removeFromCollection(res1);
      removeFromCollection(res2);
#endif
      free(res1);
      free(res2);
      return FALSE;
    }
#ifdef __SPF
  removeFromCollection(res1);
  removeFromCollection(res2);
#endif
  free(res1);
  free(res2);

  /* for each  kind of node do the pattern match */
  switch (NODE_CODE(ll1))
    {
    case VAR_REF:
      if (NODE_SYMB(ll1) == NODE_SYMB(ll2))
        return TRUE;
      break;

      /* commutatif operator */
    case EQ_OP:
      if ((NODE_SYMB(ll1) == NODE_SYMB(ll2)) &&
          patternMatchExpression(NODE_OPERAND0(ll1),
                                   NODE_OPERAND1(ll2)) &&
          patternMatchExpression(NODE_OPERAND0(ll1),
                                   NODE_OPERAND1(ll2))) 
        return TRUE;
    default :
      if ((NODE_SYMB(ll1) == NODE_SYMB(ll2)) &&
          patternMatchExpression(NODE_OPERAND0(ll1),
                                    NODE_OPERAND0(ll2)) &&
          patternMatchExpression(NODE_OPERAND1(ll1),
                                   NODE_OPERAND1(ll2))) 
        return TRUE;
    }
  return FALSE;
}

#ifdef __SPF_BUILT_IN_PARSER
PTR_LLND Follow_Llnd_temp(ll, c)
#else
PTR_LLND Follow_Llnd(ll,c)
#endif
PTR_LLND ll;
int c;
{
  PTR_LLND pt;
  if(c == 2)
    {
      pt = ll;
      while(pt)
	{
	  if (NODE_OPERAND1(pt) == NULL)
	    return pt;
	  pt = NODE_OPERAND1(pt);	  
	}
    } else
      {
        pt = ll;
        while(pt)
          {
            if (NODE_OPERAND1(pt) == NULL)
              return NODE_OPERAND0(pt);
            pt = NODE_OPERAND1(pt);	  
          }
      }
  return NULL;
}

