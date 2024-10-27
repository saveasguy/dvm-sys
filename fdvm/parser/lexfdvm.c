
/*********************************************************************/
/*                  pC++/Sage++  Copyright (C) 1993                  */
/*  Indiana University  University of Oregon  University of Rennes   */
/*********************************************************************/

/****************************************************************************

DESCRIPTION:	This is the Cedar Fortran lexical scanner.  It reads an input
		Cedar Fortran program and returns lexemes to the Cedar 
		Fortran Preprocessor.  It is based on the Unix 4.2 BSD f77
		lexical scanner.
----------------------------------------------------------------------------
CALLING FORMAT: yylex() 
----------------------------------------------------------------------------
ARGUMENTS:	<none>
----------------------------------------------------------------------------
OUTPUTS:	<return param>: integer lexeme value 
----------------------------------------------------------------------------
SIDE EFFECTS:	1].  A listing will be written to standard output based on
		     the value of global variable "yylisting".
		2].  The variable "yylineno" will be changed to reflect the
		     current line number.
		3].  The variables "yytext" and "yyleng" will be changed to
		     hold the lexeme text and its length. 
----------------------------------------------------------------------------
EXTERNAL REFS:	1].  file: <strings.h>
		2].  file: <sys/file.h>
		3].  file: <stdio.h>
		4].  file: <ctype.h>
		5].  file: defs.h
		6].  file: extern.h
		7].  file: tokdefs.h
		8].  extern int	 debug; ! when debug=0: D in col 1 is comment
----------------------------------------------------------------------------
INVOKED BY:	1].  yyparse()
----------------------------------------------------------------------------
GLOBALS:	1].  char yytext[];	! lexeme text
		2].  int  yyleng;	! length of lexeme text (in yytext)
		3].  int  yylineno;	! current line number
		4].  int  yyi4;		! =4 default/ =2 if OPTIONS/NOI4 used
		5].  int  yylisting;	! when yylisting=1: listing on stdout
		6].  int  yylonglines;	! when yylonglines=0: cols >73 comment
---------------------------------------------------------------------------- */

#include <string.h>
/*#include <sys/file.h>*/
#include <stdio.h>
#include <ctype.h>
#include <memory.h>
#include "defs.h"
#include "extern.h"
#include "tokdefs.h"

void fatal();
void fatali();
void fatalstr();
void errstr_fatal();
void err();
void err_line();
void warn_line();
char* chkalloc();
int getcd();
int getcd_ff();
int _filbuf();
int randomio();
int eqn();
int getkwd();
FILE* open_include_file();
void warn();
void free();
void make_file_list();
static void flush_comments();
static void store_comment();
static void store_comment_end_permbuff();
static void move_comments_to_permbuffer();

/* Scanner constants */

#define XEOF		0
#define HPF             1
/* Character definitions */

#define BLANKC		' '
#define MYQUOTE		(2)
#define MYHOLLERITH	(3)
#define BADHOLLERITH	(4)
#define NEWLINE		10

/* input Fortran card types */

#define STEOF		1
#define STINITIAL	2
#define STCONTINUE	3
#define CMT             4

/* lexical scanner states */

#define NEWSTMT		1
#define FIRSTTOKEN	2
#define OTHERTOKEN	3
#define RETEOS		4
#define COMMENTS	5
#define COMMENTSEOS     6
#define OVER            7

/* Declared limits */

#define MAX_NAME_LEN		80	/* max length of Fortran name */
#define MAXINCLUDES		8	/* max number of include levels */
#define MAXPARLEV		100	/* max parenthesis nesting level */
#define MAX_CONTIN_CARDS	1000	/* max number of continuation lines */
#define NCHARS_PER_LINE		66	/* max chars/line without longlines */
#define BUFFERSIZE		1024	/* number bytes read in one read */
#define COMMENT_LENGTH          1000    /* Size of comment buffer */
#define LONG_LINE_LENGTH        132

/* is_equal added for FORTRAN M */
#define is_equal(x) (((x)=='='))

#ifdef __SPF_BUILT_IN_PARSER
#define HPF_program HPF_program_
#define OMP_program OMP_program_
#define ACC_program ACC_program_
#endif

extern FILE *outf;
extern char   *infname;
 
extern int only_debug;  /*main program  sets it*/

extern int dvm_debug;     /*set to 1 by -d1 ...-d4 flags */ /*OMP*/

extern int HPF_program;
extern int OMP_program; /*OMP*/
extern int ACC_program; /*ACC*/
extern int SAPFOR;      /*SPF*/
extern int free_form;
extern int ftn_std;
extern int d_line;
extern int nchars_per_line; /* -extend_source option permits fixed form source lines to contain up to 132 characters */
                            /*  nchars_per_line = extend_source ? extend_source - 6 : NCHARS_PER_LINE */

/* Variables shared with calling routine */    

extern int     yylisting ;	/* flag: =1: make listing on stdout */
extern int     yylonglines;	/* flag: =0: col >72 for comments */
extern int     yyleng;		/* length of text in yytext */
extern char     yyquote;	/*first symbol of string constant:  " or ' */
int     yylineno;		/* source line num : current lexeme */
extern long    yystno;		/* statement number */
extern PTR_FNAME the_file;      /*pointer to the current include file*/
extern int  data_stat;          /* flag: =1: inside data statement*/

/* Globals */

static FILE *infile;    	/* file descriptor num: current file */
static char *nextch;		/* next to be considered in stmtbuf */
static char *lastch;		/* last char in this stmt in stmtbuf */
static char *nextcd = (char *)NULL;	/* place for next card in stmtbuf */
static char *endcd;		/* current end-of-card in stmtbuf */
static char *newname = (char *)NULL;	/* pointer to string */

static int restcd = 0;		/* rest-of-card  */
static int char_cntx = 0;       /* 2, inside the character constant "...."
                                   1, inside the character constant '....'
                                   0, outside the character constant */ 
static int lineno;
static int prevlin;
static int thislin;
static int stkey;		/* lexeme for overall type of stmt */
static int lastend = 1;
static int parlev;		/* current parenthesis level */
static int lexstate = NEWSTMT;	/* state within yylex */
static int nincl = 0;		/* current number of include levels */

/* Globals for: getcds/getcd/list_line/is_include/xgetc/unxgetc */

static int code = 0;		/* result of getcd (STEOF,STINITIAL,STCONTINUE) */
static char prefix[6];		/* 6 chars of Fortran stmt prefix */
static char *prefixend = prefix + 6;	/* end of prefix */
static int lines_returned = 0;	/* num lines read in getcd */
static int is_directive_hpf= 0; /* 0 - isn't directive HPF */
                                /* 1 - is directive HPF(state within getcd) */
static int is_directive_omp=0;  /* 0 - isn't directive OpenMP */
                                /* 1 - is directive OpenMP(state within getcd) */
static int is_directive_acc=0;  /* 0 - isn't directive ACC */
                                /* 1 - is directive ACC(state within getcd) */
static int is_directive_spf=0;  /* 0 - isn't directive SPF */
                                /* 1 - is directive SPF(state within getcd) */
static int end_file ;           /* 0 - doesn't appear EOF in statement line(card)  */
                                /* 1 - appears EOF in statement line(card)(state within getcd)*/
int statement_kind = 0;         /* 0 - is a fortran statement */     
                                /* 1 - is directive HPF/DVM(state within getcds) */
                                /* used in ftn.gram */
int is_acc_statement= 0;        /* 0 - is not a acc-directive */     
                                /* 1 - is acc-directive (state within getcds) */
int is_spf_statement= 0;        /* 0 - is not a spf-directive */     
                                /* 1 - is spf-directive (state within getcds) */
                                 
int directive_key;              /* key of DVM directive == stkey */
static int has_continuation = 0; /* 1 - there is symbol & in string */

/* Globals for: crunch/analyz */

static int expcom;		/* flag: =1: exposed , found */
static int expeql;		/* flag: =1: exposed = found */
static int expdcolon;           /* flag: =1; exposed :: found */
static int expscolon;           /* flag: =1; exposed : found */
static int expeqlt;             /* flag: =1; exposed => found */

/* Globals for gettok */
/*static int operator_slash;*//*OMP*/

/* Statement numbers and the statement buffer */

static long int stno;		/* stmt number for this Fortran stmt */
static long int nxtstno;	/* stmt number for next Fortran stmt */
char stmtbuf[LONG_LINE_LENGTH * (MAX_CONTIN_CARDS + 1)];
				/* this Fortran stmt + 1 more card */
char *commentbuf;
static char tempbuf[COMMENT_LENGTH];
static char *stmtend = stmtbuf + (MAX_CONTIN_CARDS * LONG_LINE_LENGTH);
				/* max addr for this Fortran stmt */
static char crunchbuf[LONG_LINE_LENGTH * (MAX_CONTIN_CARDS + 1)];
/* "Crunch"ed statement is placed in crunchbuf so that unmodified stmtbuf
    is avialable to the parser                                             */
#define COMMENT_BUF_STORE 162    /*17.10.16 podd 161=>162 */

typedef struct comment_buf {
	struct comment_buf *next;
	char *last;
	char buf[COMMENT_BUF_STORE];
	} comment_buf;
static struct comment_buf *cbfirst, *cblast, *pcbfirst, *pcblast;

/* Comment buffering data

	Comments are kept in a list until the statement before them has
   been parsed.  This list is implemented with the above comment_buf
   structure and the pointers cbnext and cblast.

	The comments are stored with terminating NULL, and no other
   intervening space.  The last few bytes of each block are likely to
   remain unused.
*/


/* INCLUDE file information block */

struct Inclfile {
	struct Inclfile *inclnext;/* pointer to next higher block */
	FILE   *inclfp;		/* file descriptor number for file */
	char   *inclname;	/* filename of this file */
	int     incllno;
	char   *incllinp;	/* point to saved nextcd when doing include */
	int     incllen;	/* length of the saved card */
	int     inclcode;	/* saved status when read nextcd */
	int     inclstno;	/* saved statement number */
	char   *prefix;		/* save columns 1-6 for later listing */
        int     ishpf;          /* has 'CHPF$' or 'CDVM$'prefix */ 
        int     rest;           /* length of card rest to be read */
        int     context;        /* saved value of variable char_cntx*/
        PTR_FNAME file_name;    /* point to structure 'file_name' of this file*/
        comment_buf *cbfirst;   /* point to next comment: pcbfirst */ 
        comment_buf *cblast;    /* point to next comment: pcblast */

};

struct Inclfile *inclp = (struct Inclfile *)NULL;	/* pointer to current INCLUDE block */


/* Valid punctuation */

struct Punctlist {
	char    punchar;
	int     punval;
} puncts[] = {
	{'(', LEFTPAR},
	{'<', LEFTAB},
	{'>', RIGHTAB},
	{')', RIGHTPAR},
	{'=', EQUAL},
	{',', COMMA},
	{'+', PLUS},
	{'-', MINUS},
	{'*', ASTER},
	{'&', AMPERSAND},
	{'/', SLASH},
	{':', COLON},
	{'\'', QUOTE},
	{'"', DQUOTE},
        {'%', PERCENT},
        {'[', LEFTAB},
        {']', RIGHTAB},
        {'@', AT},  
	{0, 0}
};


/* Valid relational operators */

struct Dotlist {
	char   *dotname;
	int     dotval;
} dots[] = {
	{"and.", AND},
	{"or.", OR},
	{"xor.", XOR},
	{"not.", NOT},
	{"true.", TTRUE},
	{"false.", FFALSE},
	{"eq.", EQ},
	{"ne.", NE},
	{"lt.", LT},
	{"le.", LE},
	{"gt.", GT},
	{"ge.", GE},
	{"neqv.", NEQV},
	{"eqv.", EQV},
	{0, 0}
};


/* Lists of keywords:  Keylist for statement keywords */

struct Keylist {
	char   *keyname;
	int     keyval;
};
struct Keylist *keystart[26], *keyend[26];	/* ptrs: start/end places for
						 * each letter of alphabet */

struct Keylist keys[] = {
/*	{"accept", ACCEPT},*/
        {"across", ACROSS},
	{"alignwith", ALIGN_WITH},
	{"align", ALIGN},
        {"allocatable", ALLOCATABLE},
        {"allocate", ALLOCATE},
        {"and", AND},
        {"apply_fragment", SPF_APPLY_FRAGMENT},
        {"apply_region", SPF_APPLY_REGION},
	{"assignment", ASSIGNMENT},
	{"assign", ASSIGN},
	{"async", ACC_ASYNC},   /*ACC*/
	{"backspace", BACKSPACE},
/*      {"block_cyclic", BLOCK_CYCLIC}, */
	{"blockdata", BLOCKDATA},
        {"block", BLOCK}, 
	{"byte", BYTE},
        {"by", BY},
	{"call", CALL},
        {"casedefault", DEFAULT_CASE},	
	{"case", CASE},
/*        {"channel", CHANNEL},*/
	{"character", CHARACTER},
/*	{"cdoacross", CDOACROSS}, */
/*	{"cdoall", CDOALL}, */
/*        {"clear", CLEAR},*/
        {"close", CLOSE},
	{"cluster", CLUSTER},
        {"code_coverage", SPF_CODE_COVERAGE},
	{"common", COMMON},
	{"complex", COMPLEX},
/*	{"concurrent", CONCURRENT},*/
        {"consistent", CONSISTENT_SPEC},
	{"contains", CONTAINS},
	{"continue", CONTINUE},
        {"copyin", OMPDVM_COPYIN},/*OMP*/
        {"copyprivate", OMPDVM_COPYPRIVATE},/*OMP*/
        {"corner", CORNER},
        {"cover", SPF_COVER},   /*SPF*/
/*        {"criticalsection", CRITICALSECTION},*/
        {"cuda_block", ACC_CUDA_BLOCK},   /*ACC*/
        {"cuda", ACC_CUDA},   /*ACC*/
	{"cycle", CYCLE},
	{"data", DATA},
        {"deallocate", DEALLOCATE},
/*	{ "decode", DECODE }, */
/*      {"decomposition", DECOMPOSITION}, */
	{"default", DEFAULT_CASE},
	{"define", DEFINE},
        {"derived", DERIVED},
/*        {"diag", DIAG}, */
	{"dimension", DIMENSION},
	{"distribute", DISTRIBUTE},
/*	{"doacross", DOACROSS}, */
	{"doubleprecision", DOUBLEPRECISION},
	{"doublecomplex", DOUBLECOMPLEX},
	{"dowhile", DOWHILE},
	{"do", DOWHILE},        
        {"dynamic", DYNAMIC},
        {"elemental", ELEMENTAL},
	{"elseif", ELSEIF},
	{"elsewhere", ELSEWHERE},
	{"else", ELSE},
/*        {"empty", EMPTY}, */
/*	{ "encode",  ENCODE }, */
/*	{"endcdoacross", ENDCDOACROSS},*/
/*	{"endcdoall", ENDCDOALL},*/
/*        {"endchannel", ENDCHANNEL},*/
/*!!!	{"endcloop", ENDCLOOP},*/
/*!!!	{"endconcurrent", ENDCONCURRENT},*/
/* !!!        {"endcriticalsection", ENDCRITICALSECTION},*/
        {"endblockdata", ENDUNIT},
/*	{"enddoacross", ENDDOACROSS},*/
	{"enddo", ENDDO},
/* !!!        {"endextended", ENDEXTENDED},*/
/* !!!        {"endextend", ENDEXTEND},*/
	{"endfile", ENDFILE},
        {"endforall", ENDFORALL},
        {"endfunction", ENDUNIT},
	{"endif", ENDIF},
        {"endinterface", ENDINTERFACE},
        {"endmodule", ENDUNIT},
        {"endprogram", ENDUNIT},
/* !!!        {"endparalleldo", ENDPARALLELDO},
              {"endparallelsections", ENDPARALLELSECTIONS},
              {"endparallel", ENDPARALLEL}, 
              {"endpdo", ENDPDO},
*/
/*!!!        {"endprocessdo", ENDPROCESSDO},*/
/*!!!        {"endprocesses", ENDPROCESSES},*/
/* !!!        {"endpsections", ENDPSECTIONS},*/
/*!!!	{"endsdoall", ENDSDOALL},*/
/* !!!        {"endsections", ENDSECTIONS},*/
        {"endselect", ENDSELECT},
/* !!!        {"endsingleprocess", ENDSINGLEPROCESS},*/
        {"endsubroutine", ENDUNIT},
        {"endtype", ENDTYPE},
	{"endwhere", ENDWHERE},
	{"end", ENDUNIT},
	{"entry", ENTRY},
	{"equivalence", EQUIVALENCE},
        {"eqv", EQV},
        {"err", ERR},
/*!!!        {"event", EVENT},*/
        {"except", SPF_EXCEPT}, /*SPF*/
        {"exit", EXIT},
/*        {"extended", EXTENDED},*/
/*        {"extend", EXTEND},*/
	{"external", EXTERNAL},
	{"expand", SPF_EXPAND }, /*SPF*/
        {"files_count",SPF_FILES_COUNT}, /*SPF*/
        {"files",FILES},
	{"find", FIND},
        {"fission",SPF_FISSION}, /*SPF*/
        {"firstprivate", OMPDVM_FIRSTPRIVATE},/*OMP*/
        {"flexible", SPF_FLEXIBLE}, /*SPF*/
        {"forall", FORALL},
	{"format", FORMAT},
/*        {"from", FROM}, */
	{"function", FUNCTION},
        {"gate", GATE},
        {"gen_block", GEN_BLOCK},
	{"global", GLOBAL_A},
	{"goto", PLAINGOTO},
        {"guided", OMPDVM_GUIDED},/*OMP*/
/*        {"guards", GUARDS},*/
        {"high", HIGH},
        {"host", ACC_HOST},   /*ACC*/     
        {"if", OMPDVM_IF}, /*OMP*/
	{"implicitnone", IMPLICITNONE},
	{"implicit", IMPLICIT},
	{"include_to", INCLUDE_TO},
	{"include", INCLUDE},
        {"indirect_access",INDIRECT_ACCESS},
        {"indirect",INDIRECT},
	{"inlocal", ACC_INLOCAL},  /*ACC*/
	{"inout", INOUT},
/*        {"inport", INPORT},*/
	{"inquire", INQUIRE},
	{"integer", INTEGER},
        {"intent", INTENT},
	{"interfaceassignment", INTERFACEASSIGNMENT},
	{"interfaceoperator", INTERFACEOPERATOR},
	{"interface", INTERFACE},
        {"interval", SPF_INTERVAL}, /*SPF*/
	{"intrinsic", INTRINSIC},
	{"in", IN},
	{"iand", IAND},/*OMP*/
	{"ieor", IEOR},/*OMP*/
	{"ior", IOR},/*OMP*/
        {"iostat", IOSTAT},
        {"iter", SPF_ITER}, /*SPF*/
        {"lastprivate", OMPDVM_LASTPRIVATE},/*OMP*/
/*        {"location", LOCATION},*/
/*        {"lock", LOCK},*/
	{"logical", LOGICAL},
/*	{"localize", LOCALIZE},*/
	{"local", ACC_LOCAL},  /*ACC*/
	{"loop", LOOP},
        {"low",LOW},
        {"maxloc", MAXLOC},
/*        {"maxparallel", MAXPARALLEL},*/
        {"max", MAX},
        {"merge", SPF_MERGE}, /*SPF*/
        {"minloc", MINLOC},
        {"min", MIN},
        {"moduleprocedure", MODULE_PROCEDURE},
	{"module", MODULE},
	{"mult_block", MULT_BLOCK},
/*        {"moveport", MOVEPORT},*/
	{"namelist", NAMELIST},
        {"neqv", NEQV},
        {"new_value", NEW_VALUE},
        {"new", NEW},
        {"noinline", SPF_NOINLINE}, /*SPF*/
        {"nowait", OMPDVM_NOWAIT},/*OMP*/
        {"none", OMPDVM_NONE},/*OMP*/
        {"nullify", NULLIFY},
        {"num_threads", OMPDVM_NUM_THREADS},/*OMP*/
	{"only", ONLY},
	{"onto", ONTO},
        {"on", ON},
	{"open", OPEN},
	{"operator", OPERATOR},
        {"optional", OPTIONAL},
	{"ordered", OMPDVM_ORDERED}, /*OMP*/
        {"or", OR},
	{"otherwise", OTHERWISE},
/*        {"outport", OUTPORT},*/
	{"out", OUT},
/*	{"overflow", OVERFLOW},*/
/* !!!        {"paralleldo", PARALLELDO},
 *        {"parallelsections", PARALLELSECTIONS},
*/
        {"parallel", PARALLEL},
	{"parameter", PARAMETER},
	{"pause", PAUSE},
/*        {"pdo", PDO},*/
        {"pointer", POINTER},
/*        {"port", PORT},*/
/*       {"post", POST}, */
	{"print", PRINT},
        {"private", PRIVATE}, 
/*!!!        {"probe", PROBE}, */
/*!!!        {"procedure", PROCEDURE},*/
        {"process_private", SPF_PROCESS_PRIVATE}, /*SPF*/
        {"product", PRODUCT},
	{"program", PROGRAM},
/*!!!	{"processcluster", PROCESS_CLUSTER},*/
/*!!!	{"processcommon", PROCESS_COMMON},*/
/*!!!        {"processdo", PROCESSDO},*/
/*!!!        {"processes", PROCESSES},*/
/*!!!	{"processglobal", PROCESS_GLOBAL},*/
        {"processors", HPF_PROCESSORS},
/*!!!        {"process", PROCESS},*/
/*        {"psections", PSECTIONS},*/
        {"public", PUBLIC},
        {"pure", PURE},
	{"range", RANGE},
	{"read", READ},
	{"real", REAL},
/*        {"receive", RECEIVE},*/
	{"recursive", RECURSIVE},
/*	{"reduce", REDUCE}, */ 
        {"reduction", REDUCTION},
/*        {"release", RELEASE},*/
        {"region", ACC_REGION}, /*ACC*/
        {"remote_access", REMOTE_ACCESS_SPEC},
        {"result", RESULT},
	{"return", RETURN},
	{"rewind", REWIND},
        {"runtime", OMPDVM_RUNTIME},/*OMP*/
	{"save", SAVE},
/*!!!        {"scommon", SCOMMON},
 *     	{"sdoall", SDOALL}, 
 *      {"sectionwait", SECTION},
 */
 	{"section", SECTION},
	{"select", SELECT},
/*        {"send", SEND},*/
        {"sequence", SEQUENCE},
/*        {"set", SET},*/
        {"schedule", OMPDVM_SCHEDULE},/*OMP*/
/*        {"shadow_add", SHADOW_ADD}, */
        {"shadow_compute", SHADOW_COMPUTE},
        {"shadow_start", SHADOW_START_SPEC},
        {"shadow_wait", SHADOW_WAIT_SPEC},
        {"shadow_renew", SHADOW_RENEW},
        {"shadow", SHADOW},
        {"shared", OMPDVM_SHARED},/*OMP*/
/*        {"singleprocess", SINGLEPROCESS},*/
/*        {"skippasteof", SKIPPASTEOF},*/
        {"shrink",SPF_SHRINK}, /*SPF*/
        {"stage", STAGE},
        {"static", STATIC},
        {"status", STATUS}, 
        {"stat", STAT}, 
	{"stop", STOP},
/*        {"submachine", SUBMACHINE},*/
	{"subroutine", SUBROUTINE},
        {"sum",SUM},
	{"sync", SYNC},
/*	{"taskcluster", TASK_CLUSTER},*/
/*      {"taskglobal", TASK_GLOBAL},*/
        {"targets", ACC_TARGETS},  /*ACC*/
        {"target", TARGET},
        {"template", HPF_TEMPLATE},
        {"tie", ACC_TIE},    /*ACC*/
        {"time", SPF_TIME}, /*SPF*/
	{"then", THEN},
	{"to", TO},
	{"type", TYPE},
/*        {"unlock", UNLOCK},*/
        {"unroll",SPF_UNROLL},              
	{"use", USE},
        {"varlist", VARLIST},
	{"virtual", VIRTUAL},
        {"wait", WAIT},
        {"wgt_block", WGT_BLOCK},
	{"where", WHERE},
	{"while", WHILE},
        {"with", WITH},
/*	{"wrap", WRAP},*/
	{"write", WRITE},
	{0, 0}
};


/*
 * This routine closes all files opened in this session
 */
void
close_files()
{
	register struct Inclfile *p = inclp;
	while (p) {
		(void)fclose(p->inclfp);
		p = p->inclnext;
	}
}


/*
 * This routine initializes the search tables for the keyword search
 * and the parameter keyword search.
 */
void
initkey()
{
	register struct Keylist *p;
	register int i, j;

	for (i = 0; i < 26; ++i)
		keystart[i] = (struct Keylist *)NULL;

	for (p = keys; p->keyname; ++p) {
		j = p->keyname[0] - 'a';
		if (keystart[j] == (struct Keylist *)NULL)
			keystart[j] = p;
		keyend[j] = p;
	}
}

int 
inilex(name)
	char   *name;		/* filename to be scanned  */
{
	void    doinclude();

	initkey();
	cbfirst = NULL;
	cblast = NULL;
	pcbfirst = NULL;
	pcblast = NULL;
	nincl = 0;
	inclp = NULL;
        infile = stdin;
	doinclude(name);
	lexstate = NEWSTMT;
	return (NO);
}

int /*OMP*/
dbginilex(name, debug)
	char   *name;		/* filename to be scanned  */
	char   *debug;		/* filename to be scanned  */
{
	void dodbginclude();
	initkey();
	cbfirst = NULL;
	cblast = NULL;
	pcbfirst = NULL;
	pcblast = NULL;
	nincl = 0;
	inclp = NULL;
        infile = stdin;
	dodbginclude(name, debug);
	lexstate = NEWSTMT;
	return (NO);
}


/* throw away the rest of the current line */
void 
flline()
{
	lexstate = RETEOS;
}

void 
doinclude(name)
	char   *name;		/* file name string of the INCLUDE file to be
				 * set up */
{
	FILE   *fp;
	struct Inclfile *t;

	if (inclp) {
	        /* --lines_returned; */ /*podd 18.04.99*/
		inclp->incllno = lines_returned;   /*podd 18.04.99*/
		/*inclp->incllno = thislin;*/
                inclp->inclcode = code;
                inclp->inclstno = stno;
                inclp->ishpf = is_directive_hpf;
                if(is_directive_omp)
                  inclp->ishpf+=2; 
                else if(is_directive_spf)
                  inclp->ishpf+=4; 
                else if(is_directive_acc)
                  inclp->ishpf+=8; 
                inclp->rest = restcd;
                inclp->context = char_cntx;
                inclp->cbfirst = cbfirst; 
                inclp->cblast  = cblast;
		if(nextcd && !free_form) {
			inclp->incllinp = copyn(inclp->incllen = endcd-nextcd , nextcd);
                        inclp->prefix = copyn(6,prefix);
                }  
		else
			inclp->incllinp = 0;
	}
	nextcd = (char *)NULL;     
        lines_returned = 0;  /*podd 18.04.99*/
        restcd = 0;
        char_cntx = 0;
        has_continuation = 0;
        cbfirst = NULL;
        cblast  = NULL;
	if(++nincl >= MAXINCLUDES)
		fatal("includes nested too deep\n");
	if (name[0] == '\0')
		fp = stdin;
	else 
            if(!inclp)    
		fp = fopen(name, "r"); /*source file*/
            else {
                fp = open_include_file(name); /*include file*/
                make_file_list(name); /*podd 18.04.99*/
            }
	if (fp)
        {
		t = inclp;
		inclp = (struct Inclfile *) chkalloc(sizeof(struct Inclfile));
		inclp->inclnext = t;
		prevlin = thislin = 0;
		inclp->inclname = name;
		infname = copys(name);
		infile = inclp->inclfp = fp;
                the_file =  inclp->file_name = cur_thread_file; /*podd 18.04.99*/
	} else {
           if(inclp)
	     fatalstr("Can not open file %s", name, 5);
           else
             errstr_fatal("Can not open file %s", name, 5);
        } 
}

void /*OMP*/
dodbginclude(name, debug)
	char   *name;		/* file name string of the INCLUDE file to be set up */
	char   *debug;		/* file name string of the INCLUDE file to be set up */

{
	FILE *fp = NULL;
	struct Inclfile *t;
	nextcd = (char *)NULL;     
        lines_returned = 0;  /*podd 18.04.99*/
        restcd = 0;
        char_cntx = 0;
        has_continuation = 0;
	if (name[0] == '\0')
		fp = stdin;
	else 
            if(!inclp)    
		fp = fopen(debug, "r"); /*source file*/
	if (fp)
        {
		t = inclp;
		inclp = (struct Inclfile *) chkalloc(sizeof(struct Inclfile));
		inclp->inclnext = t;
		prevlin = thislin = 0;
		inclp->inclname = name;
		infname = copys(name);
		infile = inclp->inclfp = fp;
                the_file =  inclp->file_name = cur_thread_file; /*podd 18.04.99*/
	} else {
           if(inclp)
	     fatalstr("Can not open file %s", debug, 5);
           else
             errstr_fatal("Can not open file %s", debug, 5);
        } 
} /*OMP*/

/*
 * This routine eliminates the current INCLUDE file block and sets up the
 * immediately previous INCLUDE file block as the current block.
 *
 *	return value = YES (1) if we are at INCLUDE level >= 2
 *		       NO  (0) if we are at the top level (we hit end-of-file)
 */
int 
popinclude()
{
	struct Inclfile *t;
	register char *p,*pp;
	register int k;
	void clf();

/* Free the space for this include file's buffer. */

	if (infile != stdin)
		clf(&infile);
	free(infname);

	--nincl;
	t = inclp->inclnext;
	free((char *)inclp->inclname);
	free((char *) inclp);
	inclp = t;
	if (!inclp)
		return (NO);

	infile = inclp->inclfp;
	infname = copys(inclp->inclname);
	prevlin = thislin = inclp->incllno;
        lines_returned =  inclp->incllno;  /*podd 18.04.99*/
        the_file = inclp->file_name;  /*podd 18.04.99*/
	code = inclp->inclcode;
	stno = nxtstno = inclp->inclstno;
        is_directive_hpf = inclp->ishpf ? 1 : 0;
        is_directive_acc = is_directive_spf = is_directive_omp = 0;
        if( inclp->ishpf >=8 )
           is_directive_acc = 1;
        else if(inclp->ishpf >=4)
           is_directive_spf = 1;
        else if(inclp->ishpf >=2)
           is_directive_omp = 1;
        restcd = inclp->rest;
        char_cntx = inclp->context;
        cbfirst = inclp->cbfirst;
        cblast  = inclp->cblast; 
	if (inclp->incllinp && !free_form) {
		nextcd = ""; /*OMP*/
		endcd = nextcd = stmtbuf;
		k = inclp->incllen;
		p = inclp->incllinp;
		while (--k >= 0)
			*endcd++ = *p++;
		free((char *) (inclp->incllinp));
                k = 6;
                p = inclp->prefix;
                pp = prefix;
                while (--k >= 0)
			*pp++ = *p++;
		free((char *) (inclp->prefix));
	} else
		nextcd = (char *)NULL;
        end_file = 0;   
        has_continuation = 0;
	return (YES);
}


/*
 * This routine does one line of a listing on stdout.
 *
 *   ptr    (INPUT) -	a pointer to the body of the line to be listed (col 7)
 *   line_number (INPUT) - the line number for this line
 */
void
list_line(ptr, line_number)
	char   *ptr;
	int     line_number;
{
	if (ptr)
		(void)fprintf(outf,"%6d %c %c%c%c%c%c%c%s\n",
		       line_number, ((nincl > 1) ? 'I' : ' '),
		       prefix[0], prefix[1], prefix[2], prefix[3], prefix[4], prefix[5], ptr);
}


/*
 * This routine fetches all the cards necessary to get a complete Fortran
 * statement.  One additional card after the full Fortran statement must 
 * be read to determine that it is not a continuation card.
 *
 *	return value - STINITIAL (a Fortran statement is ready)
 *		       STEOF	 (end of entire program)
 */
int
getcds()
{
	register char *p, *q;
	void err();
       

/* First, record how many bytes and lines have been read from the file */

first:	yylineno = lines_returned;
	if (yylisting && (code != STEOF) )	/* LISTING */
		list_line(nextcd, yylineno);

	if (newname) {
		free(infname);
		infname = newname;
		newname = (char *)NULL;
	}
	move_comments_to_permbuffer();
top:
	if (!nextcd) {
		nextcd = ""; /*OMP*/
		code = getcd(nextcd = stmtbuf);
		yylineno = lines_returned;

		if (yylisting && (code != STEOF)) /* LISTING */
			(void) list_line(nextcd, yylineno);

		stno = nxtstno;
		if (newname) {
			free(infname);
			infname = newname;
			newname = (char *)NULL;
		}
		prevlin = thislin;
	}
	if (code == STEOF) {
    	        move_comments_to_permbuffer();
		if (popinclude()) {
		        /* ++lines_returned; */  /*podd 18.04.99*/
			goto first;
		} else
			return (STEOF);
	}
	if (code == STCONTINUE) {
		if (newname) {
			free(infname);
			infname = newname;
			newname = (char *)NULL;
		}
		lineno = thislin;
		err("Illegal continuation line ignored", 9);    /* podd 21.05.14 card=>line */
		nextcd = (char *)NULL;
		goto top;
	}
	if (code == CMT) {
		nextcd = (char *)NULL;
		flline();
		return (CMT);
	}
	if (nextcd > stmtbuf) {
		for (q = nextcd, p = stmtbuf; q < endcd; *p++ = *q++)
			;
		endcd = p;
	}
	move_comments_to_permbuffer();
        statement_kind = is_directive_hpf;
	is_openmp_stmt = is_directive_omp; /*OMP*/
	is_acc_statement=is_directive_acc; /*ACC*/
	is_spf_statement=is_directive_spf; /*SPF*/
        if(end_file) {
       	   nextcd = ""; /*OMP*/
           nextcd = endcd;
           code = STEOF;
           goto end;
        } 
	nextcd = ""; /*OMP*/
	for (nextcd = endcd;
	     (nextcd+nchars_per_line <= stmtend) && ((code=getcd(nextcd)) == STCONTINUE);
	     nextcd = endcd-1)   /*podd 12.11.17: NCHARS_PER_LINE=>nchars_per_line*/
	{
       
             if(statement_kind != is_directive_hpf || is_openmp_stmt != is_directive_omp || is_acc_statement != is_directive_acc || is_spf_statement != is_directive_spf) 
             {  if (newname) {
                    free(infname);
                    infname = newname;
                    newname = (char *) NULL;
                }
                lineno = thislin;
                err_line("Illegal continuation line ignored", 9, lines_returned);

                endcd = nextcd + 1; 
                continue;
             }
	     memmove(nextcd-1,nextcd,endcd-nextcd);   /*memcpy => memmove*/  /*podd 23.12.13*/ 
	     move_comments_to_permbuffer();
		if (yylisting && (code != STEOF) && (code != CMT)) /* LISTING */
			list_line(nextcd-1, lines_returned);
             if(end_file) {
                code = STEOF;
                nextcd = endcd-1;
                break;
             } 
	}
end:	nextch = stmtbuf;
	lastch = nextcd - 1;
	if (nextcd >= stmtend)
		nextcd = (char *)NULL;
	lineno = prevlin;
	prevlin = thislin;
	return (STINITIAL);
}


/*
 * This routine fetches all the cards necessary to get a complete Fortran
 * statement (free format).
 *
 *	return value - STINITIAL (a Fortran statement is ready)
 *		       STEOF	 (end of entire program)
 */
int
getcds_ff()
{
  /*	register char *p, *q;*/
	void err(),err_line();
       

/* First, record how many bytes and lines have been read from the file */

/*first:	yylineno = lines_returned;
	if (yylisting && (code != STEOF) )	 
		list_line(nextcd, yylineno);

	move_comments_to_permbuffer();*/
top:
		nextcd = ""; /*OMP*/
		code = getcd_ff(nextcd = stmtbuf);
		yylineno = lines_returned;

		if (yylisting && (code != STEOF)) /* LISTING */
			(void) list_line(nextcd, yylineno);

		stno = nxtstno;

/*		prevlin = thislin; */
	
	if (code == STEOF) {
    	        move_comments_to_permbuffer();
		if (popinclude()) 		     
			goto top;
		else
			return (STEOF);
	}
	if (code == STCONTINUE) {
		lineno = thislin;
		err("Illegal continuation line ignored", 9);
		nextcd = (char *)NULL;
		goto top;
	}
/*	if (code == CMT) {
		nextcd = (char *)NULL;
		flline();
		return (CMT);
	}
*/

	move_comments_to_permbuffer();
        statement_kind = is_directive_hpf;
	is_openmp_stmt = is_directive_omp; /*OMP*/
	is_acc_statement=is_directive_acc; /*ACC*/
	is_spf_statement=is_directive_spf; /*SPF*/
        if(end_file) {
	   nextcd = ""; /*OMP*/
           nextcd = endcd;
           code = STEOF;
           goto end;
        }
        for(nextcd=endcd; (has_continuation != 0); nextcd=endcd) { 
             
             code=getcd_ff(nextcd);	     
       
             if(statement_kind != is_directive_hpf || is_openmp_stmt != is_directive_omp || is_acc_statement != is_directive_acc || is_spf_statement != is_directive_spf) /*ACC*/  
             {  lineno = thislin;
                err_line("Illegal continuation line ignored", 9, lines_returned );
                endcd = nextcd + 1; 
                continue;
             }

	     move_comments_to_permbuffer();

		if (yylisting && (code != STEOF) && (code != CMT)) /* LISTING */
			list_line(nextcd-1, lines_returned);

             if(end_file) {
                code = STEOF;
                nextcd=endcd;
                break;
             } 
	}
end:	nextch = stmtbuf;
	lastch = nextcd - 1;
	/*if (nextcd >= stmtend)
		nextcd = (char *)NULL;*/
	lineno = prevlin;
	prevlin = thislin;
	return (STINITIAL);
}




/*
 * This routine retrieves one card from the Fortran source file.
 *
 *	b (INPUT) -	The address at which to put the new card
 *
 *	return value - STINITIAL (the first card of a Fortran statement)
 *		       STCONTINUE (a continuation card)
 *		       STEOF (end-of-file on the current file)
 */
int 
getcd(b)
	register char *b;
{
	register int c;
	register char *p, *bend, *cmt;
	/*extern int debug;*/
	int     cnext;  /*, has_comment_sym; */
	void err(), err_line(), warn_line(), store_comment();

top:
	endcd = b;
        is_directive_hpf = 0;
        is_directive_acc = 0; /*ACC*/
        is_directive_spf = 0; /*SPF*/
        is_directive_omp = 0; /*OMP*/
        end_file = 0;
        /* bend = yylonglines? stmtend : b + NCHARS_PER_LINE */
        if(restcd) { /* reading the rest of card (after ';' )  */
         /*(void)fprintf(stderr,"getcd: restcd  = %d\n",restcd);*/
	  bend = b + restcd;
          p = prefix;
          while( p < prefixend )
            *p++ = BLANKC;
          restcd = 0;
          c = BLANKC; 
          -- lines_returned;         
          goto body;  
        }
	bend = b + nchars_per_line; /*podd 12.11.17: NCHARS_PER_LINE=>nchars_per_line*/

	/* Get first character on a line */
        c = getc(infile);  

        if (c == 'c' || c == 'C' || c == '*' || c == '!' || c == '#' 
		  || (d_line ? 0 : (c == 'D') || (c == 'd') )  ) { /* comment card */
		++lines_returned;

		/* For the listing:  fill the prefix array from the comment */

		prefix[0] = (c != 'D' && c != 'd') ? '!' : c; /* c */
		for (p = prefix + 1; p < prefixend; p++)
			if ((c = getc(infile)) == '\n')
				break;
			else if (c == EOF)
				break;
			else
				*p = c;
                if ((prefix[1]=='$') && ((prefix[2]==' ') || isdigit(prefix[2]))) { /*OMP*/
                    if(OMP_program==1 ) {   
			for (p = prefix; p < prefixend; p++) {/*OMP*/
				*p = ' ';/*OMP*/
				(void)ungetc(' ', infile);/*OMP*/
			}/*OMP*/
                        is_directive_omp = 1; /*OMP*/
			--lines_returned; /*OMP*/
			goto top;/*OMP*/
                    } else if(SAPFOR==0) {                      
                        warn_line("Ignoring line with OpenMP conditional compilation sentinel ($)", 664, lines_returned);                        
                        prefix[1] ='!';
                    }
		} 	
		/*
		 * If we saw EOF or '\n' and quit early above,
		 * fill the rest of the prefix with blanks. 
		 */

		for (; p < prefixend; p++)
			*p = ' ';

		
                if((HPF_program && eqn(4,prefix+1,"hpf$")) ||
                  (!HPF_program && !ftn_std && eqn(4,prefix+1,"dvm$"))) {                  
                    is_directive_hpf = 1;
                   -- lines_returned; 
                    goto body;
                }
                if(eqn(4,prefix+1,"$omp")) {/*OMP*/
                  if(OMP_program==1) { 
                    is_directive_hpf = 1; /*OMP*/
                    is_directive_omp = 1; /*OMP*/
                    -- lines_returned;  /*OMP*/
                    goto body; /*OMP*/
                  } else if(SAPFOR==0) {                      
                    warn_line("Ignoring OpenMP derective", 664, lines_returned);                        
                    prefix[1] = '!';
                    } 
                } /*OMP*/
                if((ACC_program==1) && (eqn(4,prefix+1,"$acc")|| eqn(4,prefix+1,"$apm"))) {/*ACC*/
                    is_directive_hpf = 1; /*ACC*/
                    is_directive_acc = 1; /*ACC*/
                    -- lines_returned;  /*ACC*/
                    goto body; /*ACC*/
                } /*ACC*/
                if((SAPFOR==1) && eqn(4,prefix+1,"$spf")) {/*SPF*/
                    is_directive_hpf = 1; /*SPF*/
                    is_directive_spf = 1; /*SPF*/
                    -- lines_returned;  /*SPF*/
                    goto body; /*SPF*/
                } /*SPF*/

                /*
                if (only_debug && (eqn(4,prefix+1,"dvm$")) && isspace(prefix[5])){
                    while((endcd < bend) && ((c = getc(infile))!='\n') && (c!=EOF) && isspace(c))
                        *endcd++=c;
                  
		    if(c=='p' || c=='P') {
                      *endcd++=c;
                      is_directive_hpf = 1;
                      -- lines_returned; 
                      goto body;
                    }
                    if ((c != EOF) && (c != '\n'))
                      *endcd++=c;   
                }
               if (only_debug  &&  statement_kind == HPF &&  (eqn(4,prefix+1,"dvm$")) && !isspace(prefix[5]))
                  {  is_directive_hpf = 1;
                     -- lines_returned; 
                     goto body;
                   }
		*/
          
		/* Now, begin consuming the rest of the comment line- */

		if ((c != EOF) && (c != '\n'))
			while (((c = getc(infile)) != '\n') &&
			       (c != EOF)) 
				*endcd++ = c;
		/*if (c == '\n') *endcd++ = c;    podd 24.11.00*/
                *endcd++ = '\n'; /* podd 24.11.00*/
		*endcd = '\0';

		if (yylisting)	/* LISTING */ 
			(void) list_line(b, lines_returned);
		for (cmt = tempbuf, p = prefix; p < prefixend; )
		  *cmt++ = *p++;
		for (p = b; *p;)
		  *cmt++ = *p++;
		*cmt = '\0';
		store_comment();
		goto top;
        }  else if (c == EOF) 
		    return (STEOF);
	         
	   else {                       /*if(c != EOF)*/
		/* a tab in columns 1-6 skips to column 7 */
		(void)ungetc(c, infile);
		for (p = prefix; p < prefixend &&
		     ((c = getc(infile)) != '\n') && (c != EOF);)
			if (c == '\t') {
				while (p < prefixend)
					*p++ = BLANKC;

			/*
			 * check out next char: if digit, it goes in col 6 of
			 * the card 
			 */

				if (isdigit(cnext = getc(infile)))
					prefix[5] = cnext;
				else
					(void)ungetc(cnext, infile);
				/*bend = stmtend;*/ /* podd 01.12.02*/
			} else
				*p++ = c;
	}
body:/* if(newname) {
               free(infname);
               infname = newname;
               newname=(char *)NULL;
        }
        lineno=thislin;
        if(is_directive_hpf) err("directive HPF");
      */
        /* has_comment_sym = 0; */
        if ((c == EOF) || (c == '\n')) {
  
	         /*return (STEOF);
		  *}
	          *if (c == '\n') {
                  */
		while (p < prefixend)
			*p++ = BLANKC;
		*endcd = '\0';	/* EOLN:  make statement empty */
                if(c == EOF)   
                       end_file = 1;
                
	} else {		/* Read body of line */
                if (is_directive_hpf) { 
                   if (isspace(prefix[5])) /*first card of directive*/
                         char_cntx = 0;
                }
                else  if ((prefix[5] == ' ') || (prefix[5] == '0')) /*first card of statement*/
                         char_cntx = 0;

		while ((endcd < bend) && ((c = getc(infile)) != '\n') && (c != EOF) )
		{        if((c=='!') && (char_cntx==0)) break;
                         if((c==';') && (char_cntx==0)) break;
                         *endcd++ = c; 
                        /*  if( c=='!') has_comment_sym=1; */
                          
                         if(c=='\'') {
                           if(char_cntx == 0)
                               char_cntx = 1;
                           else if(char_cntx == 1)
                               char_cntx = 0;
                         }
                         if(c=='\"') {
                           if(char_cntx == 0)
                               char_cntx = 2;
                           else if(char_cntx == 2)
                               char_cntx = 0;
                         }
          
                }
			
                if((c == '\n') && (endcd > b) && (*(endcd-1) == '\r')) /*podd 03.06.01*/
                  *(endcd-1)='\0'; /* deleting '\r' before '\n'*/
                                  /*('\r\n' is end line marker in Windows) */
                else if ((c == '\n') && (endcd == b) && (prefix[5]=='\r')) /*podd 07.06.02*/
                     { prefix[5]=BLANKC; *endcd++ = '\0';} 
                else
		  *endcd++ = '\0';  /* put NULL char in buffer as end marker */
                if((c==';') && (char_cntx==0)) {              
                  restcd = bend - endcd;
                  while( ((c = getc(infile)) == ';') || (c == ' '))
                    restcd--;
		  (void) ungetc(c,infile);                  		  
                }
                else if((c=='!') && (char_cntx==0)) {
                    cmt = tempbuf;
                    *cmt++ = '!';
                    while (((c = getc(infile)) != '\n') && (c != EOF)) 
				*cmt++ = c;
                    *cmt++ = '\n'; 
		    *cmt = '\0';
                    store_comment();
                }
		else if (c == EOF) 
		    /*return (STEOF);*/
                    end_file = 1;
		else if (c != '\n') {
			while ((c = getc(infile)) != '\n')
				if (c == EOF)           
                             { /*return (STEOF);*/
                               end_file = 1;
                               break;
			     }
		}

	}

	++lines_returned;

	/* Check for a comment char in the prefix */
 
        if (!is_directive_hpf) {
	for (p = prefix; p < prefixend-1; ++p)
		if (*p == '!') {
			if (yylisting)
				(void) list_line(b, lines_returned);
			for (cmt = tempbuf; p < prefixend; )
			  *cmt++ = *p++;
		        for (p = b; *p;)
		          *cmt++ = *p++;
                        *cmt++ = '\n';
                        *cmt = '\0';
/*!!!!*/
			/*(void)fprintf(stderr,"getcd:tempbuf =  %s \n",tempbuf);*/
			store_comment();
			goto top; 
		}
         }

        if (is_directive_hpf) {
            if(isspace(prefix[5]) == 0 ) /*((prefix[5] != ' ') && (prefix[5] != '0'))*/
		return (STCONTINUE);
            goto check; 
        }
	if ((prefix[5] != ' ') && (prefix[5] != '0'))
		return (STCONTINUE);
	for (p = prefix; p < prefixend; ++p)
		if (isspace(*p) == 0)
			goto initline;
check:	/* Check for blank line */
	for (p = b; p < endcd - 1; ++p)	/* char at endcd-1 is always \0 */
             if (isspace(*p) == 0) {
			nxtstno = 0;
			return (STINITIAL);
		}


	/* We have a completely blank line */

	if (yylisting)	/* LISTING */
		(void) list_line(b, lines_returned);
	goto top;

initline:			/* there is a label */
	nxtstno = 0;

 /*
  * Start out conversion loop at second char in prefix if we have debug set and
  * a D/d in col 1.  Otherwise, start at first. 
  */

	if (d_line) {
		if (prefix[0] == 'd' || prefix[0] == 'D')
			p = prefix + 1;
		else
			p = prefix;
        }
	while (p < prefix + 5) {
	        if (isspace(*p) == 0) {
		        if (isdigit(*p))
				nxtstno = 10 * nxtstno + (*p - '0');
			else {
				if (newname) {
					free(infname);
					infname = newname;
					newname = (char *)NULL;
				}
				lineno = thislin;
				err_line("Non digit in statement number field", 11, lines_returned);
				nxtstno = 0;
				break;
			}
	        }
		p++;
	}
	return (STINITIAL);
}

int 
getcd_ff(b)
	register char *b;
{
	register int c;
	register char *p,  *cmt;
	/*extern int debug;*/
	int   continuation_state, is_continuation, has_label, numsym;
	void err(), err_line(), warn_line(), store_comment();

top:
	endcd = b;
        is_directive_hpf = 0;
        is_directive_acc = 0; /*ACC*/
        is_directive_spf = 0; /*SPF*/
        is_directive_omp = 0; /*OMP*/
        is_continuation = 0;
        has_label = 0;
        end_file = 0;
        numsym = 0;        

        if(restcd) { /* reading the rest of card (after ';' )  */
          restcd = 0;
          numsym = 1; 
         /* -- lines_returned;*/                 
        } else
          lines_returned++; 
        
        if(has_continuation) {
            continuation_state = 1;
            has_continuation = 0;
        } else
            continuation_state = 0; 


	/* Get first nonblank character on a line */  
        while(isspace(c=getc(infile)) && (c != '\n'))  
          numsym++;
        
        
        if((c =='!') || (numsym == 0 && (c == '$'))) {
		/*++lines_returned;*/

		prefix[0] = c;
		for (p = prefix + 1; p < prefixend; p++)
			if ((c = getc(infile)) == '\n')
				break;
			else if (c == EOF)
				break;
			else
				*p = c;

		/*
		 * If we saw EOF or '\n' and quit early above,
		 * fill the rest of the prefix with blanks. 
		 */

		for (; p < prefixend; p++)
			*p = ' ';
                if ((prefix[1]=='$') && ((prefix[2]==' ') || (prefix[2]=='&'))) { /*OMP*/ 
                    if (OMP_program == 1) {  /*OMP*/   
			for (p = prefix; p < prefixend; p++) {/*OMP*/
				*p = ' ';/*OMP*/
				(void)ungetc(' ', infile);/*OMP*/
			}/*OMP*/
                        is_directive_omp = 1; /*OMP*/
			--lines_returned; /*OMP*/
			goto top;/*OMP*/
                    } else if(SAPFOR==0) {   /*OMP*/                   
                        warn_line("Ignoring line with OpenMP conditional compilation sentinel ($)", 664, lines_returned);  /*OMP*/                      
                        prefix[1] ='!';   /*OMP*/
                    }
		} /*OMP*/		
                if((HPF_program && eqn(4,prefix+1,"hpf$") && (prefix[0] != '$')) ||
                  (!HPF_program && !ftn_std && eqn(4,prefix+1,"dvm$") && (prefix[0] != '$'))) {
                    is_directive_hpf = 1;
                    goto body;
                } else if((ACC_program==1) && (eqn(4,prefix+1,"$acc") || eqn(4,prefix+1,"$apm"))) {/*ACC*/
                    is_directive_hpf = 1; /*ACC*/
                    is_directive_acc = 1; /*ACC*/
                    goto body; /*ACC*/ 
                } else if((SAPFOR==1) && eqn(4,prefix+1,"$spf")) {/*SPF*/
                    is_directive_hpf = 1; /*SPF*/
                    is_directive_spf = 1; /*SPF*/
                    goto body; /*SPF*/                
                } else if (eqn(4,prefix+1,"$omp")) {/*OMP*/
                    if(OMP_program == 1) {  /*OMP*/  
                       is_directive_hpf = 1; /*OMP*/
                       is_directive_omp = 1; /*OMP podd*/                  
                       goto body; /*OMP*/
                    } else if(SAPFOR==0) {  /*OMP*/
                       warn_line("Ignoring OpenMP derective", 664, lines_returned); /*OMP*/                       
                       prefix[1] = '!'; /*OMP*/                      
                    } /*OMP*/
                }  

                for(cmt = tempbuf, p=prefix; p<prefixend;)
                    *cmt++ = *p++;                
        
		/* Now, begin consuming the rest of the comment line- */

		if ((c != EOF) && (c != '\n'))
			while (((c = getc(infile)) != '\n') &&
			       (c != EOF)) 
				*cmt++ = c;
		
                *cmt++ = '\n'; 
		*cmt = '\0';		
		store_comment();
                if(c == EOF)
                   return (STEOF);
		goto top;
        }  else if (c == EOF) 
		return (STEOF);
           else if (c == '\n') {/* We have a completely blank line */
                /*++lines_returned;*/
                goto top;
        }
           else if (c == '&') {
                is_continuation = 1;
                goto body;      
        }
          else if (!continuation_state && isdigit(c)) {              
              while(isdigit(c)) {
                has_label = 10*has_label + (c - '0');
                c = getc(infile);
              }
              if(!isspace(c)) 
                 err_line("No blank after label", 330, lines_returned );
               else
                 while(isspace(c) && (c != '\n'))
                     c = getc(infile);
        }
	    
        (void) ungetc(c,infile);
body:         	/* Read body of line */
	while ((endcd < stmtend) && ((c = getc(infile)) != '\n') && (c != EOF) )
		{        if((c=='!') && (char_cntx==0)) break;
                         if((c==';') && (char_cntx==0)) break;
                         *endcd++ = c;                         
                          
                         if(c=='\'') {
                           if(char_cntx == 0)
                               char_cntx = 1;
                           else if(char_cntx == 1)
                               char_cntx = 0;
                         }
                         if(c=='\"') {
                           if(char_cntx == 0)
                               char_cntx = 2;
                           else if(char_cntx == 2)
                               char_cntx = 0;
                         }          
                }
			
                if((c == '\n') && (endcd > b) && (*(endcd-1) == '\r')) /*podd 03.06.01*/
                  *(endcd-1)='\0'; /* deleting '\r' before '\n'*/
                                  /*('\r\n' is end line marker in Windows) */                
                else
		  *endcd++ = '\0';  /* put NULL char in buffer as end marker */
                if((c==';') && (char_cntx==0)) {
                  restcd = 1;                                                    
                  while( ((c = getc(infile)) == ';') || (c == ' '))
                     ;
		  (void) ungetc(c,infile);                  		  
                }
                else if((c=='!') && (char_cntx==0)) {
                    cmt = tempbuf;
                    *cmt++ = '!';
                    while (((c = getc(infile)) != '\n') && (c != EOF)) 
				*cmt++ = c;
                    *cmt++ = '\n'; 
		    *cmt = '\0';
                    store_comment();
                    if (c == EOF) end_file = 1;
                }
		else if (c == EOF) 
		    /*return (STEOF);*/
                    end_file = 1;
		else if (c != '\n') {
			while ((c = getc(infile)) != '\n')
				if (c == EOF)           
                             { /*return (STEOF);*/
                               end_file = 1;
                               break;
			     }
		}

	/*++lines_returned;*/
        for(p=endcd-2; isspace(*p) && p>=b ;endcd-- ){/*deleting last blanks in line*/
           *p-- = '\0';
        }
        
        if( *(p) == '&') 
          {has_continuation = 1;  endcd = p;}

        if (is_directive_hpf) {
            if(prefix[5] == '&' ) 
		return (STCONTINUE);            
	/* Check for blank line */        
	  for (p = b; p < endcd - 1; ++p)	/* char at endcd-1 is always \0 */
		  if (isspace(*p) == 0) {    
		    nxtstno = 0;		 
			return (STINITIAL);  
		  }	
	  goto top;
          
        }

	if (is_continuation)
		return (STCONTINUE);

	nxtstno = has_label;

	return (STINITIAL);
}


          

/*
 * This routine removes all blanks, reduces everything to lower case
 *   (except char strings), replaces outer quotes with MYQUOTE, encloses
 *   a Hollerith string with MYHOLLERITH, and determines whether this
 *   statement contains either an exposed comma, or an exposed equal.
 */
void
crunch()
{
	register char *i, *j, *j0, *j1, *prvstr, *cmt;
	int     bracklev;
	int     ten, nh, quote;
	void    erri(), err();

/* i is the next input character to be looked at.
   j is the next output character */

	--lastch;		/* at this point, lastch is pointing at the
				 * ending '\0' decrement it to eliminate that
				 * '\0' from consideration */

	bracklev = 0;		/* level of <,> pairs */
	parlev = 0;		/* level of (,) pairs */

	expcom = 0;		/* exposed ','s */
	expeql = 0;		/* exposed '='s */
	expdcolon = 0;          /* exposed '::'s */
	expscolon = 0;          /* exposed ':' eg. construct_name : .... */
	expeqlt = 0;
	operator_slash = 0;
	prvstr = j = crunchbuf;
	memset(crunchbuf, 0, sizeof(crunchbuf)); /*OMP*/
	for (i = stmtbuf; i <= lastch; ++i) {
		if (isspace(*i) || (*i == '\0'))
			continue;
	/* eliminate ! comments	 */
		if (*i == '!') {
		     
/*			*j++ = '!'; why? */
			cmt = tempbuf;
			while (*i)
				*cmt++ = *i++;
                        *cmt++ = '\n';
			*cmt = '\0';
/*!!!*/
			/* (void)fprintf(stderr,"crunch:tempbuf = %s \n",tempbuf);*/
			store_comment_end_permbuff();
			continue;
		}
	/* Is this a QUOTE and is this not a random I/O operator? */
        /* character constant "<string>" will be change by '<string>' */
        /* "acb""d" will be change by 'abc"d' */

		if ((*i == '\'') || (*i == '\"'))   /* && (!randomio(j)))*/
                {
			quote = *i;
			*j = MYQUOTE;	/* special marker */
                        *++j = *i;
			for (;;) {
				if (++i > lastch) {
					err("Unbalanced quotes; closing quote supplied", 12);
					break;
				}
				if (*i == quote)
				{  if (i < lastch && i[1] == quote) { 
                                                *++j = *i;          
						++i;
                                   }
				   else
						break;
				}		
				*++j = *i;
			}
			if (i > lastch)
				++j;	/* quotes were unbalanced */
			else {
				j[1] = MYQUOTE;
				j += 2;
			}
			prvstr = j;


		} else if ((*i == 'h' || *i == 'H') && j > prvstr) {/*	 test for Hollerith
								     *	  strings */
			if (isdigit(j[-1])==0)
				goto copychar;
			nh = j[-1] - '0';
			ten = 10;
			j1 = prvstr - 1;
			if (j1 < j - 5)
				j1 = j - 5;
			for (j0 = j - 2; j0 > j1; --j0) {
				if (isdigit(*j0)==0)
					break;
				nh += ten * (*j0 - '0');
				ten *= 10;
			}
			if (j0 <= j1)
				goto copychar;
             
/*
 * A hollerith must be preceded by a punctuation mark.
 * '*' is possible only as repetition factor in a data statement
 * not, in particular, in character*2h
 */
	           
			if (!(*j0 == '*' && stmtbuf[0] == 'd' && stmtbuf[1] == 'a') &&
			    *j0 != '/' && *j0 != '(' &&
			    *j0 != ',' && *j0 != '=' && *j0 != '.')
				goto copychar;
			if (i + nh > lastch) {
				erri("%dH too big", nh, 326);
				nh = lastch - i;
			}
			/* Form the string for the Hollerith constant.*/ 
			j0[1] = MYHOLLERITH;	/* special marker */
			j = j0 + 1;
			while (nh-- > 0)
				*++j = *++i;
			j[1] = MYHOLLERITH;
			j += 2;
			prvstr = j;


		} else {
			if (*i == '(') {
				if (++parlev > MAXPARLEV)
					fatal("Too many levels of parenthesis nesting");
			} else if (*i == ')')
				--parlev;
/*			else if (*i == '<')
				++bracklev;
			else if (*i == '>')
				--bracklev;
			else if ((*i == ':') && (i[1] == ':') && (bracklev == 0))
  			        expdcolon = 1;
*/			else if ((parlev == 0) && (bracklev == 0))
                        {       if ((*i == ':') && (i[1] == ':')) 
				{
				     *j++ = *i++;
				     expdcolon = 1;
				}
			        else if (*i == ':')
				     expscolon = 1;
				else if ((*i == '=') && (i[1] == '>') && (expdcolon == 0))
				{
				     *j++ = *i++;
				     expeqlt = 1;
				}
				else if ((*i == '=') && (i[1] == '='))
				     *j++ = *i++;
				else if ((*i == '=') && (expdcolon == 0))
				     expeql = 1;
				else if (*i == ',')
				     expcom = 1;
			}	     
	copychar:		/* not a string or space -- copy, shifting case
				 * if necessary */
			*j++ = (shiftcase && isupper(*i)) ? tolower(*i) : *i;
		}
	}
	{ register char *k;
	for (i = k = stmtbuf; i <= lastch; ++i)
            if (*i != '\0') { *k = *i; ++k; }
        }
	lastch = j - 1;
 	nextch = crunchbuf;
}


/*
 * This routine is only called when the current character in crunch is 
 * a single quote (').  It determines whether the beginning of the current
 * statement looks like the beginning of a random I/O statement.  The 
 * parameter j is a pointer to the next place where a 'crunch'ed character
 * will be put within crunchbuf.  The characters under scrutiny are:
 * crunchbuf[0] through j[-1].
 *
 *      j (INPUT) - the next place within crunchbuf where a 'crunch'ed
 *                  character will be placed.
 *
 *	return value - YES if the chars in cruncbuf look like a random I/O
 *                     stmt NO  otherwise
 */
int 
randomio(j)
	register char *j;
{
	int     nchars;
	char   *start;
	register char *p;

	nchars = j - crunchbuf;	/* compute number of characters in buffer */

	if ((nchars > 6) && eqn(6, crunchbuf, "write("))
		start = crunchbuf + 6;
	else if ((nchars > 5) && eqn(5, crunchbuf, "read("))
		start = crunchbuf + 5;
	else if ((nchars > 5) && eqn(5, crunchbuf, "find("))
		start = crunchbuf + 5;
	else
		return (NO);

	if (isalpha(*start))
	/* We have an identifier for a unit number.  Rest must be alphanum-- */
	{
		for (p = start + 1; p < j; ++p)
			if ((isalpha(*p)==0) && (isdigit(*p)==0) && !(*p == '_'))
				return (NO);
		return (YES);
	} else if (isdigit(*start))
	/* We have an integer constant for a unit number. */
	{
		for (p = start + 1; p < j; ++p)
			if (isdigit(*p) == 0)
				return (NO);
		return (YES);
	} else
		return (NO);
}


int 
named_stmt()
{
     char *i, *p;
     i = nextch;
     p = yytext;
     *p++ = *i++;
     while (i <= lastch)
	  if (isalpha(*i) || isdigit(*i) || (*i == '_'))
	       *p++ = *i++;
	  else
	       break;
     yyleng = p - yytext;
     *p = '\0';
     if ((*i == ':') && (i[1] != ':') && (i != nextch)) 
     {
	  nextch = i;
	  return (1);
     }
     else return (0);
}

	  
/*
 * This routine determines the type of statement we have.  It either
 * finds the first lexeme, (like FORMAT, LOGICALIF, etc), or determines
 * that this is an assignment statement (LET).  It assigns that value
 * to the variable "stkey" which stays around as a reminder of what
 * type statement we have as the rest of the statement is scanned.
 */
void
analyz()
{
	register char *i;
	int forallstmt, level;
        int hpf;
	void err();

        directive_key = 0;
        hpf = 0;
	if (parlev) {
		err("Unbalanced parentheses, statement skipped", 13);
		stkey = UNKNOWN;
		return;
	}

	if ((expscolon) && (named_stmt()))
	{
	     stkey = CONSTRUCT_ID;
	     return;
	}
/* !!! */
        if (statement_kind != HPF) {	
	if (nextch + 2 <= lastch && eqn(3, nextch, "if(")) {
               /* assignment or if statement -- look at character after balancing paren */
		parlev = 1;
		for (i = nextch + 3; i <= lastch; ++i)
			if (*i == (MYQUOTE))
				while (*++i != MYQUOTE);
			else if (*i == (MYHOLLERITH))
				while (*++i != MYHOLLERITH);
			else if (*i == '(')
				++parlev;
			else if (*i == ')') {
				if (--parlev == 0)
					break;
			}
		if (i >= lastch)
			stkey = LOGICALIF;
		else if (i[1] == '=')
			stkey = LET;
		else if (isdigit(i[1]))
			stkey = ARITHIF;
		else
			stkey = LOGICALIF;
		if (stkey != LET)
			nextch += 2;
	} else if (nextch + 5 <= lastch && eqn(6, nextch, "where(")) {
               /* assignment or where statement -- look at character after balancing paren */
		parlev = 1;
		for (i = nextch + 6; i <= lastch; ++i)
			if (*i == (MYQUOTE))
				while (*++i != MYQUOTE);
			else if (*i == (MYHOLLERITH))
				while (*++i != MYHOLLERITH);
			else if (*i == '(')
				++parlev;
			else if (*i == ')') {
				if (--parlev == 0)
					break;
			}
		if (i >= lastch)
			stkey = WHERE;
		else if (i[1] == '=')
			stkey = LET;
		else
			stkey = WHERE_ASSIGN;
		if (stkey != LET)
			nextch += 5;
	} else if (expeql) {	/* may be an assignment */
		if (expcom && nextch < lastch) {
			if ((nextch[0] == 'd') && (nextch[1] == 'o')) {
				stkey = PLAINDO;
				nextch += 2;
/*!!!
                        } else if (eqn(9, nextch, "processdo")) {
                                stkey = PROCESSDO;
                                nextch += 9;

			} else if (eqn(10, nextch, "paralleldo")) {
				stkey = PARALLELDO;
				nextch += 10;
			} else if (eqn(3, nextch, "pdo")) {
				stkey = PDO;
				nextch += 3;

			} else if (eqn(5, nextch, "alldo")) {
				stkey = CDOALL;
				nextch += 5;
			} else if (eqn(8, nextch, "acrossdo")) {
				stkey = CDOACROSS;
				nextch += 8;
			} else if (eqn(6, nextch, "cdoall")) {
				stkey = CDOALL;
				nextch += 6;
			} else if (eqn(6, nextch, "sdoall")) {
				stkey = SDOALL;
				nextch += 6;
			} else if (eqn(8, nextch, "doacross")) {
				stkey = DOACROSS;
				nextch += 8;
			} else if (eqn(9, nextch, "cdoacross")) {
				stkey = CDOACROSS;
				nextch += 6;
			} else if (eqn(6, nextch, "forall")) {
				stkey = FORALLDO;
				nextch += 6;
!!!*/

			} else
				stkey = LET;
		}

		else if (eqn(7, nextch, "forall(")) {
			level = 0;
			forallstmt = -1;
			for (i = nextch + 7; i<= lastch; i++) {
				switch (*i) {
				    case (')'):
					if (level <= 0)
						if (*(i + 1) == '=') {
							forallstmt = NO;
							break;
						} else {
							forallstmt = YES;
							break;
						}
					else
						level--;
					break;
				    case ('('):
					level++;
				    default:
					break;
				}
				if (forallstmt >= 0) {
					if (forallstmt == 1) {
						stkey = FORALL;
						nextch += 6;
					} else
						stkey = LET;
					break;
				}
			}
			if (i > lastch)
				stkey = LET;
		}

                 else
			stkey = LET;
	}
	else if (expeqlt && !expcom)
	     stkey = POINTERLET;
/* otherwise search for keyword */
	else {
        
                      /*printf("analyz: %d\n",yylineno);*/
  		stkey = getkwd();
 		if ((stkey == TYPE) && (nextch[0] != '('))
 		     stkey = TYPE_DECL;
 		if (stkey == PLAINGOTO && lastch >= nextch){
 		     if (nextch[0] == '(')
			  stkey = COMPGOTO;
		     else if (isalpha(nextch[0]))
			  stkey = ASSIGNGOTO;                
                }
                if (stkey == ELSEIF && nextch[0] != '('){
                          stkey = ELSE;
                          nextch -= 2;          
                }              
	}
	parlev = 0;
        }
        else  { /* for HPF, DVM, OMP and SPF directives */
          if(is_spf_statement) {  
             if ( eqn(8,  nextch,   "analysis")) {
                stkey = SPF_ANALYSIS;   
                nextch += 8;
             }  else if (eqn(15, nextch, "endparallel_reg")) {
                stkey = SPF_END_PARALLEL_REG;    
                nextch += 15;
             }  else if (eqn(12, nextch, "parallel_reg")) {
                stkey = SPF_PARALLEL_REG;    
                nextch += 12;
             }  else if (eqn(8, nextch, "parallel")) {
                stkey = SPF_PARALLEL;  
                nextch += 8;
             }  else if (eqn(9, nextch, "transform")) {
                stkey = SPF_TRANSFORM;    
                nextch += 9;
             }  else if (eqn(10, nextch, "checkpoint")) {
                stkey = SPF_CHECKPOINT;    
                nextch += 10;

             }  else
                stkey = UNKNOWN;
             parlev = 0;
             return;
          } 
          switch(*nextch) {
	  case 's':
             if ( eqn(12,  nextch,   "shadow_group")) {
                stkey = SHADOW_GROUP;   
                nextch += 12;
             }  else if (eqn(12, nextch, "shadow_start")) {
                stkey = SHADOW_START;  
                nextch += 12;
             }  else if (eqn(11, nextch, "shadow_wait")) {
                stkey = SHADOW_WAIT;    
                nextch += 11;
             }  else if (eqn(10, nextch, "shadow_add")) {
                stkey = SHADOW_ADD;    
                nextch += 10;
             }  else if (eqn(6, nextch, "shadow")) {
                stkey = SHADOW;       
                nextch += 6;
	     }  else if (eqn(8, nextch, "sections")) { /*OMP*/
                stkey = OMPDVM_SECTIONS;
                nextch += 8;
             }  else if (eqn(7, nextch, "section")) { /*OMP*/
                stkey = OMPDVM_SECTION;
                nextch += 7;
             }  else if (eqn(6, nextch, "single")) { /*OMP*/
                stkey = OMPDVM_SINGLE;       
                nextch += 6;
             }  else
                stkey = UNKNOWN;
              break;
          case 'r':
             if  (eqn(12,  nextch,      "redistribute")) {
                stkey = REDISTRIBUTE;  hpf = 1;
                nextch += 12;
             }  else if (eqn(15, nextch, "reduction_group")) {
                stkey = REDUCTION_GROUP;  
                nextch += 15;
             }  else if (eqn(15, nextch, "reduction_start")) {
                stkey = REDUCTION_START;   
                nextch += 15;
             }  else if (eqn(14, nextch, "reduction_wait")) {
                stkey = REDUCTION_WAIT;    
                nextch += 14;
             }  else if (eqn(11, nextch, "realignwith")) {
                stkey = REALIGN_WITH;  hpf = 1;
                nextch += 11;
             }  else if (eqn(7, nextch, "realign")) {
                stkey = REALIGN;      hpf = 1;
                nextch += 7;
             }  else if (eqn(13, nextch, "remote_access")) {
                stkey = REMOTE_ACCESS;   
                nextch += 13;
             }  else if (eqn(12, nextch, "remote_group")) {
                stkey = REMOTE_GROUP;    
                nextch += 12;
             }  else if (eqn(4, nextch, "real")) {
                stkey = REAL;
                nextch += 4;
             }  else if (eqn(5, nextch, "reset")) {
                stkey = RESET;     
                nextch += 5;
             }  else if (eqn(6, nextch, "region")) {  /*ACC*/
                stkey = ACC_REGION;  
                nextch += 6;
             }  else if (eqn(7, nextch, "routine")) { /*ACC*/
                stkey = ACC_ROUTINE;  
                nextch += 7; 
            }  else
                stkey = UNKNOWN;
             break;
          case 'd':
             if ( eqn(10,  nextch,    "distribute")) {
                stkey = DISTRIBUTE;  hpf = 1;
                nextch += 10;
             }  else if (eqn(7, nextch, "dynamic")) {
                stkey = DYNAMIC;     hpf = 1;
                nextch += 7;
             }  else if (eqn(15, nextch, "doubleprecision")) {
                stkey = DOUBLEPRECISION;
                nextch += 15;
             }  else if (eqn(9, nextch, "dimension")) {
                stkey = DIMENSION;   hpf = 1;
                nextch += 9;
             }  else if (eqn(5, nextch, "debug")) {
                stkey = DEBUG;       hpf = 1;
                nextch += 5;
             }  else if (eqn(13,  nextch,  "doublecomplex")) {
                stkey = DOUBLECOMPLEX;
                nextch += 13;
             }  else if (eqn(2,  nextch,  "do")) { /*OMP*/
		if (OMP_program == 1) { /*OMP*/
	                stkey = OMPDVM_DO; /*OMP*/
        	        nextch += 2; /*OMP*/
		}/*OMP*/
             }  else
                stkey = UNKNOWN;
             break;
          case 'p': 
             if ( eqn(10,  nextch,  "processors")) {
                stkey = HPF_PROCESSORS;  hpf = 1;
                nextch += 10;
             }  else if (eqn(8, nextch, "prefetch")) {
                stkey = PREFETCH ;   
                nextch += 8;
             }  else if (eqn(10, nextch, "paralleldo")) { /*OMP*/
                stkey = OMPDVM_PARALLELDO ;   
                nextch += 10;
             }  else if (eqn(16, nextch, "parallelsections")) { /*OMP*/
                stkey = OMPDVM_PARALLELSECTIONS;   
                nextch += 16;
             }  else if (eqn(17, nextch, "parallelworkshare")) { /*OMP*/
                stkey = OMPDVM_PARALLELWORKSHARE;
                nextch += 17;
             }  else if (eqn(8, nextch, "parallel")) {
                if (!is_openmp_stmt) stkey = PARALLEL ;  /*OMP*/
		else stkey = OMPDVM_PARALLEL; /*OMP*/
                directive_key = stkey;
                nextch += 8;
             }  else
                stkey = UNKNOWN;
             break;
          case 'i':
             if ( eqn(11,  nextch,  "independent")) {
                stkey = INDEPENDENT;  hpf = 1;
                nextch += 11;
             }  else if (eqn(7, nextch, "integer")) {
                stkey = INTEGER;
                nextch += 7;
             }  else if (eqn(8, nextch, "interval")) {
                stkey = INTERVAL;     hpf = 1;
                nextch += 8;
             }  else if (eqn(7, nextch, "inherit")) {
                stkey = INHERIT;      hpf = 1;
                nextch += 7;
             }  else if (eqn(15, nextch, "indirect_access")) {
                stkey = INDIRECT_ACCESS;   
                nextch += 15;
             }  else if (eqn(14, nextch, "indirect_group")) {
                stkey = INDIRECT_GROUP;    
                nextch += 14;
             }  else if (eqn(7, nextch, "io_mode")) {
                stkey = IO_MODE;    
                nextch += 7;
             }  else
                stkey = UNKNOWN;
             break;
          case 'e':
             if (eqn(11,  nextch,   "endinterval")) {
                stkey = ENDINTERVAL;  hpf = 1;
                nextch += 11;
             }  else if (eqn(12, nextch, "exitinterval")) {
                stkey = EXITINTERVAL;     hpf = 1;
                nextch += 12;
             }  else if (eqn(8, nextch, "enddebug")) {
                stkey = ENDDEBUG;     hpf = 1;
                nextch += 8;
             }  else if (eqn(14, nextch, "endtask_region")) {
                stkey = ENDTASK_REGION;   
                nextch += 14;
             }  else if (eqn(5, nextch, "endon")) {
                stkey = ENDON;    
                nextch += 5;
             }  else if (eqn(15, nextch, "endasynchronous")) {
                stkey = ENDASYNCHRONOUS;    
                nextch += 15;
             }  else if (eqn(11, nextch, "endcritical")) {/*OMP*/
                stkey = OMPDVM_ENDCRITICAL;
	        nextch += 11;
             }  else if (eqn(11, nextch, "endsections")) {/*OMP*/
                stkey = OMPDVM_ENDSECTIONS;
	        nextch += 11;
             }  else if (eqn(9, nextch, "endmaster")) {/*OMP*/
                stkey = OMPDVM_ENDMASTER;
	        nextch += 9;
             }  else if (eqn(10, nextch, "endordered")) {/*OMP*/
                stkey = OMPDVM_ENDORDERED;
	        nextch += 10;
             }  else if (eqn(9, nextch, "endsingle")) {/*OMP*/
                stkey = OMPDVM_ENDSINGLE;
	        nextch += 9;
             }  else if (eqn(12, nextch, "endworkshare")) {/*OMP*/
                stkey = OMPDVM_ENDWORKSHARE;
	        nextch += 12;
             }  else if (eqn(5, nextch, "enddo")) {/*OMP*/
                stkey = OMPDVM_ENDDO;
	        nextch += 5;
             }  else if (eqn(20, nextch, "endparallelworkshare")) {/*OMP*/
                stkey = OMPDVM_ENDPARALLELWORKSHARE;
                nextch += 20;
             }  else if (eqn(19, nextch, "endparallelsections")) {/*OMP*/
                stkey = OMPDVM_ENDPARALLELSECTIONS;
                nextch += 19;
             }  else if (eqn(13, nextch, "endparalleldo")) {/*OMP*/
                stkey = OMPDVM_ENDPARALLELDO;
                nextch += 13;
             }  else if (eqn(11, nextch, "endparallel")) {/*OMP*/
                stkey = OMPDVM_ENDPARALLEL;    
                nextch += 11;
             }  else if (eqn(9, nextch, "endregion")) {   /*ACC*/
                stkey = ACC_END_REGION;  
                nextch += 9;
            }  else if (eqn(14, nextch, "endhostsection")) { /*ACC*/
                 stkey = ACC_END_CHECKSECTION;
                 nextch += 14;
            }  else
                stkey = UNKNOWN;
             break;
          case 'a':
             if ( eqn(9,  nextch,  "alignwith")) {
                stkey = ALIGN_WITH;  hpf = 1;
                nextch += 9;
             }  else if (eqn(5, nextch, "align")) {
                stkey = ALIGN;       hpf = 1;
                nextch += 5;
             }  else if (eqn(7, nextch, "asyncid")) {
                stkey = ASYNCID;       
                nextch += 7;
             }  else if (eqn(12, nextch, "asynchronous")) {
                stkey = ASYNCHRONOUS;      
                nextch += 12;
             }  else if (eqn(9, nextch, "asyncwait")) {
                stkey = ASYNCWAIT;       
                nextch += 9;
             }  else if (eqn(6, nextch, "atomic")) { /*OMP*/
                stkey = OMPDVM_ATOMIC;       
                nextch += 6;
             }  else if (eqn(6, nextch, "actual")) { /*ACC*/
                stkey = ACC_ACTUAL;       
                nextch += 6;
             }  else
                stkey = UNKNOWN;
             break;
          case 'c': 
             if (eqn(7,  nextch,  "complex")) {
                stkey = COMPLEX;
                nextch += 7;
             }  else if (eqn(9, nextch, "character")) {
                 stkey = CHARACTER;
                 nextch += 9;
             }  else if (eqn(5, nextch, "check")) {
                 stkey = CHECK;
                 nextch += 5;
             }  else if (eqn(16, nextch, "consistent_group")) {
                 stkey = CONSISTENT_GROUP;
                 nextch += 16;
            }  else if (eqn(16, nextch, "consistent_start")) {
                 stkey = CONSISTENT_START;
                 nextch += 16; 
            }  else if (eqn(15, nextch, "consistent_wait")) {
                 stkey = CONSISTENT_WAIT;
                 nextch += 15; 
            }  else if (eqn(10, nextch, "consistent")) {
                 stkey = CONSISTENT;
                 nextch += 10;
            }  else if (eqn(8, nextch, "critical")) { /*OMP*/
                 stkey = OMPDVM_CRITICAL;
                 nextch += 8;
            }  else if (eqn(9, nextch, "cp_create")) { 
                 stkey = CP_CREATE;
                 nextch += 9;
            }  else if (eqn(7, nextch, "cp_load")) { 
                 stkey = CP_LOAD;
                 nextch += 7;
            }  else if (eqn(7, nextch, "cp_save")) { 
                 stkey = CP_SAVE;
                 nextch += 7;
            }  else if (eqn(7, nextch, "cp_wait")) { 
                 stkey = CP_WAIT;
                 nextch += 7;
            }  else
                stkey = UNKNOWN;
             break;
	  case 't':
             if (eqn(15, nextch, "template_create")) {
                stkey = TEMPLATE_CREATE;   
                nextch += 15;
             } else if (eqn(15, nextch, "template_delete")) {
                stkey = TEMPLATE_DELETE;   
                nextch += 15;  
             } else if (eqn(8, nextch, "template")) {
                stkey = HPF_TEMPLATE;   hpf = 1;
                nextch += 8; 
             } else if (eqn(7, nextch, "traceon")) {
                stkey = TRACEON;        hpf = 1;
                nextch += 7; 
             } else if (eqn(8, nextch, "traceoff")) {
                stkey = TRACEOFF;       hpf = 1;
                nextch += 8;  
             } else if (eqn(11, nextch, "task_region")) {
                stkey = TASK_REGION;   
                nextch += 11; 
             } else if (eqn(4, nextch, "task")) {
                stkey = TASK;      
                nextch += 4;  
             } else if (eqn(13, nextch, "threadprivate")) { /*OMP*/
                stkey = OMPDVM_THREADPRIVATE;      
                nextch += 13;  
             }/* else if (eqn(11, nextch, "num_threads")) {
                stkey = OMPDVM_NUM_THREADS;      
                nextch += 11;  
             }*/  else
                stkey = UNKNOWN;
             break;
	  default:
             if (eqn(9, nextch, "new_value")) {
                stkey = NEW_VALUE;    
                nextch += 9;
             }  else if (eqn(7, nextch, "logical")) {
                stkey = LOGICAL;
                nextch += 7;   
             }  else if (eqn(8, nextch, "localize")) {
                stkey = LOCALIZE;
                nextch += 8;   
             }  else if (eqn(3, nextch, "map")) {
                stkey = MAP;     
                nextch += 3; 
             }  else if (eqn(3, nextch, "f90")) {
                stkey = F90;
                nextch += 3;       
             }  else if (eqn(5, nextch, "flush")) { /*OMP*/
                stkey = OMPDVM_FLUSH;
                nextch += 5;
             }  else if (eqn(9, nextch, "workshare")) {
                stkey = OMPDVM_WORKSHARE;
                nextch += 9;       
             }  else if (eqn(2, nextch, "on")) {
                stkey = ON_DIR;
                nextch += 2;        
             } else if (eqn(4, nextch, "heap")) {
                stkey = HEAP;
                nextch += 4; 
             } else if (eqn(7, nextch, "barrier")) {
		if (!is_openmp_stmt) stkey = BARRIER;  /*OMP*/
		else stkey = OMPDVM_BARRIER;  /*OMP*/
		hpf = 1;
                nextch += 7;          
             } else if (eqn(5, nextch, "nodes")) {
		stkey = OMPDVM_NODES;  /*OMP*/
                nextch += 5;
             } else if (eqn(13, nextch, "fromonethread")) {
                stkey = OMPDVM_ONETHREAD;
                nextch += 13;       
             } else if (eqn(6, nextch, "master")) {
                stkey = OMPDVM_MASTER;
                nextch += 6;       
             } else if (eqn(7, nextch, "ordered")) { /*OMP*/
                stkey = OMPDVM_ORDERED;
                nextch += 7; 
             }  else if (eqn(10, nextch, "get_actual")) { /*ACC*/
                stkey = ACC_GET_ACTUAL;       
                nextch += 10; 
            }  else if (eqn(11, nextch, "hostsection")) { /*ACC*/
                 stkey = ACC_CHECKSECTION;
                 nextch += 11;     
             } else
                stkey = UNKNOWN; 
             break; 
	  }
          if(HPF_program && hpf != 1) 
                stkey = UNKNOWN;   /* some FDVM directives are illegal in HPF-DVM */
          if(!HPF_program &&  stkey == INDEPENDENT)
                stkey = UNKNOWN;   /* INDEPENDENT directive is illegal in FDVM */
        }
        parlev = 0;
}


/*
 * This routine checks the characters starting at "nextch" against the
 * table of Fortran keywords.
 */
int 
getkwd()
{
	register char *i, *j;
	register struct Keylist *pk, *pend;
	int     k, kval;

        /* the following line is useful for testing parser */
          /* fprintf(stderr,"getkwd:::  %d %s\n", yylineno, nextch);*/
          /* printf("getkwd:::  %d %s\n", yylineno, nextch);*/
	if (isalpha(nextch[0])==0)
		return (LET);
	k = nextch[0] - 'a';
        pk = keystart[k];
	if (pk)
		for (pend = keyend[k]; pk <= pend; ++pk) {
			i = pk->keyname;
 /*!!!*/             /*   if(pk->keyval == ACC_USES && !is_acc_statement)
                          continue;
                     */

 /*!!!*/                if(pk->keyval == ACC_TARGETS && statement_kind != HPF)
                          continue;
 /*!!!*/                if(pk->keyval == BYTE && statement_kind == HPF)
                          continue;
 /*!!!*/                if(pk->keyval == ONLY && statement_kind == HPF)
                          continue;
 /*!!!*/                if(pk->keyval == ONTO && directive_key == PARALLEL)
                          continue;
                     
                       /* (void)fprintf(stderr,"getkwd:keyname %s\n",i);*/
                       /* (void)printf("getkwd:keyname %s\n",i);        */
			j = nextch;
			while (*++i == *++j && *i != '\0');
			if (*i == '\0' && j <= lastch + 1) {
				nextch = j;
/*!!!*/
				/*(void)fprintf(stderr,"getkwd:keyword %s = %d\n",pk->keyname, pk->keyval);*/
				/*(void)printf("getkwd:keyword %s = %d\n",pk->keyname, pk->keyval);*/
	                        
				  kval = pk->keyval;

                                  if((kval == ONLY)  && (statement_kind == HPF))
                                     return(ON);
                                  if((kval == POINTER) && (statement_kind == HPF))
                                     return(DVM_POINTER);
			          if(statement_kind != HPF)
                                  switch (kval) {
				  case HPF_TEMPLATE:
                                  case HPF_PROCESSORS:
                                  case DYNAMIC:
                                  case ALIGN:
                                  case ALIGN_WITH:
                                  case SHADOW:
                                  case DISTRIBUTE:  
                                     return (UNKNOWN);
                                  default:
                                     return (kval);
				}                                
                                 return (kval);
			}
		}
/*!!!*/
	/*(void)fprintf(stderr,"getkwd:keyword UNKNOWN\n");*/
	
	return (UNKNOWN);
}


/*
 * This routine gets the next token in a statement.  It starts at "nextch"
 * within "stmtbuf".  As a side effect, "yytext" is filled with the
 * consumed input and "yyleng" is set to the length of that text.
 *
 *	return value - the lexeme value for the token
 */
int 
gettok()
{
	register struct Dotlist *pd;
	register struct Punctlist *pp;
	/*register struct oplist *op;*/
	register char *p, *i, *j, *saveptr;

	int     havdot, havexp, havdbl;
	int     val, ijk;
	extern struct Punctlist puncts[];
	extern struct Dotlist dots[];
	/*extern struct oplist operands[];*/
	extern int yyleng;
	void err();

	char   *n1;
        int boz;

        boz = 0;
        if(data_stat && (*nextch == 'b' || *nextch == 'o' || *nextch == 'z' || *nextch == 'x') && nextch[1] == MYQUOTE) { /* 'x' is extension in Intel compiler */
	  ++nextch;
          boz = 1;
        } 
/* Check for ordinary character string: 'xxxxxxx' */
	if (*nextch == (MYQUOTE)) {
		++nextch;
		p = yytext;
                yyquote = *nextch++; 
		while ((nextch <= lastch) && (*nextch != MYQUOTE))
			*p++ = *nextch++;
		if (nextch > lastch) {
			yyleng = p - yytext;
			*p = '\0';
			return (BAD_CCONST);
		}
		++nextch;
		yyleng = p - yytext;
		*p = '\0';

                if(boz==1)
                  return (BOZ_CONSTANT);
                else
	       /* Are we sure this is a char constant? (could be hex, octal or binary, Intel compiler extension ) podd 29.12.14 */
		{ if ((*nextch != 'x') &&
		      (*nextch != 'o') &&
                      (*nextch != 'b'))
			
                     return (CHAR_CONSTANT);
                  else
                /* It's a hex, octal or binary constant! - Intel compiler  */
        	  {  
        	     ++nextch;
                     return (BOZ_CONSTANT);
                  }
                }
/*		else {
			if (*nextch == 'x')
				radix = 16;
			else
				radix = 8;

*/		/* Check for bad chars in constant. */

/*			for (p = yytext + 1; p < (yytext + (yyleng - 1));)
				if (*p == BLANKC || *p == '\t')
					p++;
				else {
					if (isupper(*p))
						*p = tolower(*p);
					if (hextoi(*p++) >= radix) {
						radix = 0;
						break;
					}
				}
			++nextch;
			return (radix == 0 ? BAD_SYMBOL :
				(radix == 16 ? HEX_CONSTANT : OCTAL_CONSTANT));
	        }
*/       }
/* Check for a Hollerith constant */

	if (*nextch == (MYHOLLERITH)) {
		++nextch;
		p = yytext;
		*p++ = '\'';	/* opening quote */
		while (*nextch != MYHOLLERITH) {
			if (*nextch == '\'')
				*p++ = '\'';	/*-explode quotes-*/
			*p++ = *nextch++;
		}
		++nextch;
		*p++ = '\'';	/* closing quote */
		yyleng = p - yytext;
		*p = '\0';
		return (CHAR_CONSTANT);
	}
/* Check for a bad Hollerith constant (as found by crunch) */

	if (*nextch == (BADHOLLERITH)) {
		++nextch;
		yytext[0] = '1';
		yytext[1] = 'h';
		yytext[yyleng = 2] = '\0';
		return (BAD_CCONST);
	}

	if (needkwd) {
		needkwd = 0;
 /*!!! */
		/*(void)fprintf(stderr,"gettok:case needkw\n");*/
		return (getkwd());
	}
        if (implkwd) {
		implkwd = 0;
                saveptr = nextch;
                ijk = getkwd();
                if(ijk == UNKNOWN)
  	           return (ijk);
                if(*nextch != '(') 
                   return (ijk);
                parlev =1;
	        for (i = nextch + 1; i <= lastch; ++i)
			if (*i == (MYQUOTE))
				while (*++i != MYQUOTE);
			else if (*i == (MYHOLLERITH))
				while (*++i != MYHOLLERITH);
			else if (*i == '(')
				++parlev;
			else if (*i == ')') {
				if (--parlev == 0)
					break;
			}

		if ((i <= lastch) &&(i[1] == '('))
                                  /* IMPLICIT statemet is of kind: IMPLICIT REAL(kind-selector)(spec-letter_list) */
		  return(ijk);
		else {            /* IMPLICIT statemet is of kind (F77): IMPLICIT REAL (spec-letter_list) */
                  nextch=saveptr;
                  needkwd=1;
                  return(STAT);
                }
	}
        if (optkwd) {
                optkwd = 0;
                saveptr = nextch;
                ijk = getkwd();
                if ((ijk != UNKNOWN) && (is_equal(*nextch)))
                    return(ijk);
                else
                    nextch = saveptr;
        }
        if (opt_kwd_) {
                opt_kwd_ = 0;
                saveptr = nextch;
                ijk = getkwd();
                if ((ijk != UNKNOWN)&&(ijk!= LET))
                    return(ijk);
                else
                    nextch = saveptr;
        }
        if (opt_kwd_r) {
                opt_kwd_r = 0;
                saveptr = nextch;
                ijk = getkwd();
                if ((ijk != UNKNOWN)&&(ijk!= LET) && (*nextch == '('))
                    return(ijk);
                else
                    nextch = saveptr;
        }
        if (as_op_kwd_) {
                as_op_kwd_ = 0;
                saveptr = nextch;
                ijk = getkwd();
                if ((ijk != UNKNOWN)&&((ijk == ASSIGNMENT) || (ijk == OPERATOR)) && (*nextch == '('))
                    return(ijk);
                else
                    nextch = saveptr;
        }
       if (opt_kwd_hedr) {
                opt_kwd_hedr = 0;
                saveptr = nextch;
                ijk = getkwd();
                if ((ijk != UNKNOWN)&&((ijk == FUNCTION) || (ijk == PURE) || (ijk == RECURSIVE) || (ijk == ELEMENTAL)))
                    return(ijk);
                else
                    nextch = saveptr;
        }
        if (optcorner) {
	  /* (void)fprintf(stderr,"gettok:case optcorner\n");*/
                optcorner = 0;
                saveptr = nextch;
                ijk = getkwd();
                if (ijk == CORNER)
                    return(ijk);
                else
                    nextch = saveptr;
        }
        if (optcall) {
                optcall = 0;
                saveptr = nextch;
                ijk = getkwd();
                if ((ijk != UNKNOWN) && (ijk == CALL))
                    return(ijk);
                else
                    nextch = saveptr;
        }
        if (opt_in_out) {
                opt_in_out = 0;
                saveptr = nextch;
                ijk = getkwd();
                if ((ijk != UNKNOWN) && ((ijk == IN) || (ijk == OUT)) && (*nextch == ':'))
                    return(ijk);
                else
                    nextch = saveptr;
        }

	/*        if (opt_l_s) {
                opt_l_s = 0;
                saveptr = nextch;
                ijk = getkwd();
                if ((ijk != UNKNOWN) && ((ijk == LOCATION) || (ijk == SUBMACHINE)))
                    return(ijk);
                else
                    nextch = saveptr;
        }
        */

/* Check for punctuation */
	for (pp = puncts; pp->punchar; ++pp)
		if (*nextch == pp->punchar) {
			if ((*nextch == '*' || *nextch == '/') &&
			    nextch < lastch && nextch[1] == nextch[0]) {
				if (*nextch == '*')
					val = DASTER;
				else
					val = DSLASH;
				nextch += 2;
			} 
			else if ((*nextch == '/') && ( nextch < lastch))
			{
			  if (nextch[1] == ')' )
			    {
                               if  (operator_slash == 1)
		               {
			          val = SLASH;
				  ++nextch;
				  operator_slash = 0;
			       }
			       else 
			       {
				  val = RIGHTAB;
				  nextch += 2;
			       }
			    }       
			  else if (nextch[1] == '=')
			      {
			         val = NE;
				 nextch += 2;
			      }       
			  else 
			     {
				  val = SLASH;
				  ++nextch;
			     }
			}
			else if ((*nextch == '(') && ( nextch < lastch))
			{
			     if (nextch[1] == '/') { 
                               if (nextch[2] == ')') {
		                  val = LEFTPAR;
				  ++nextch;
                                  ++parlev;
				  operator_slash = 1;
                               } else if (nextch[2] == '/') {
		                  val = LEFTPAR;
				  ++nextch;
                                  ++parlev;
                               } else if (nextch[2] == '=') {
		                  val = LEFTPAR;
				  ++nextch;
                                  ++parlev;
                               } else {
 	                          if (operator_slash == 1) { /*OMP*/
				     val = LEFTPAR; /*OMP*/
				     ++nextch; /*OMP*/
				     operator_slash = 0; /*OMP*/
				  } /*OMP*/ else {
                                     val = LEFTAB;
				     nextch += 2;
				  }	
                               }
			     } else {
				  val = LEFTPAR;
				  ++parlev;
				  ++nextch;
			     }
			}
			else if ((*nextch == '=') && (nextch < lastch))
			{
			     if (nextch[1] == '>')
			     {
				  val = POINT_TO;
				  nextch += 2;
			     }
			     else if (nextch[1] == '=')
			     {
				  val = EQ;
			     	  nextch += 2;
			     }
			     else 
			     {
				  val = EQUAL;
				  ++nextch;
			     }
			}
			else if ((*nextch == '<') && (nextch < lastch))
			{
			     if (nextch[1] == '>')
			     {
				  val = NE;
				  nextch += 2;
			     }
			     else if (nextch[1] == '=')
			     {
				  val = LE;
			     	  nextch += 2;
			     }
			     else
			     {
				  val = LT;
				  ++nextch;
			     }
			}
			else if ((*nextch == '>') && (nextch < lastch))
			{
			     if (nextch[1] == '=')
			     {
				  val = GE;
			     	  nextch += 2;
			     }
			     else 
			     {
				  val = GT;
				  ++nextch;
			     }
			}
			else {
				val = pp->punval;
				if (val == LEFTPAR)
					++parlev;
				else if (val == RIGHTPAR)
					--parlev;
				++nextch;
			}
			return (val);
		}
/* Check for dotted lexeme (.AND. .OR. .XOR.  ...) */

	if (*nextch == '.')
	{	if (nextch >= lastch)
			goto badchar;
		else if (isdigit(nextch[1]))
			goto numconst;
		else {
			for (pd = dots; (j = pd->dotname); ++pd) {
				for (i = nextch + 1; i <= lastch; ++i)
					if (*i != *j)
						break;
					else if (*i != '.')
						++j;
					else {
						nextch = i + 1;
						return (pd->dotval);
					}
			}
			p = yytext;
			*p++ = *nextch++;
			while ((nextch <= lastch) && (*nextch != '.'))
			     if (isalpha(*nextch))
				  *p++ = *nextch++;
			     else
				  goto badchar;
			if (*nextch != '.')
			     goto badchar;
			*p++ = *nextch++;
			yyleng = p - yytext;
                        *p = '\0';
			return (DEFINED_OPERATOR);
		}
	}	
/* Check for underscore '_' */
        if(*nextch == '_'){
          ++nextch;
	  return(UNDER);
        }
/* Check for an alphabetic lexeme */
	if (isalpha(*nextch)) {
	/* Now, match an ordinary identifier. */

		p = yytext;
		*p++ = *nextch++;
		while (nextch <= lastch)
                  if(*nextch == '_') {
                    if (nextch[1] == MYQUOTE)
                      break;
                    else
                      *p++ = *nextch++;
                  }
		  else if (isalpha(*nextch) || isdigit(*nextch) || (*nextch == '$'))
		    *p++ = *nextch++;
		  else
		    break;
		yyleng = p - yytext;
		*p = '\0';
                if (colon_flag && nextch <= lastch && (*nextch == ':') &&
                    !strcmp(yytext, "only"))
		{
		     ++nextch;
		     colon_flag = 0;
		     return (ONLY);
		}
		if (inioctl && nextch <= lastch && *nextch == '=' && nextch[1] != '=') { 
			++nextch;
			return (NAMEEQ);
		}
	/* Check to make sure that it was not a FUNCTION def */
		/*
		if (yyleng > 17 && eqn(17, yytext, "recursivefunction") &&
		    isalpha(yytext[17]) && (nextch < lastch) && (nextch[0] == '(') &&
		    ((nextch[1] == ')') || isalpha(nextch[1]))) {
			nextch -= (yyleng - 9);
			return (RECURSIVE);
		}
		if (yyleng > 8 && eqn(8, yytext, "function") &&
		    isalpha(yytext[8]) && (nextch < lastch) && (nextch[0] == '(') &&
		    ((nextch[1] == ')') || isalpha(nextch[1]))) {
			nextch -= (yyleng - 8);
			return (FUNCTION);
		}
                */
		if (yyleng > MAX_NAME_LEN) {
			char    buff[200];
			(void)sprintf(buff, "name %s too long, truncated to %d",
				yytext, MAX_NAME_LEN);
			err(buff, 170);
			/*yyleng = MAX_NAME_LEN;
			yytext[MAX_NAME_LEN] = '\0';  */
		}
		return (IDENTIFIER);
	}
	if (*nextch == '!') {
		++nextch;
	        return(COMMENT);
	}

/* Check for a numeric lexeme */

	if (isdigit(*nextch)==0)
		goto badchar;
numconst:
	havdot = NO;
	havexp = NO;
	havdbl = NO;
	for (n1 = nextch; nextch <= lastch; ++nextch) {
		if (*nextch == '.')
			if (havdot)
				break;
			else if (nextch + 2 <= lastch && isalpha(nextch[1])
				 && isalpha(nextch[2]))
				break;
			else
				havdot = YES;
		else if (!intonly && (*nextch == 'd' || *nextch == 'e')) {
			p = nextch;
			havexp = YES;
			if (*nextch == 'd')
				havdbl = YES;
			if (nextch < lastch)
				if (nextch[1] == '+' || nextch[1] == '-')
					++nextch;
			if ((nextch >= lastch) || (isdigit(*++nextch)==0)) {
				nextch = p;
				havdbl = havexp = NO;
				break;
			}
			for (++nextch;
			     nextch <= lastch && isdigit(*nextch);
			     ++nextch);
			break;
		} else if (isdigit(*nextch)==0)
			break;
	}
	p = yytext;
	i = n1;
	while (i < nextch)
		*p++ = *i++;
	yyleng = p - yytext;
	*p = '\0';
	if (havdbl)
		return (DP_CONSTANT);
	if (havdot || havexp)
		return (REAL_CONSTANT);

/* Have an INTEGER constant.  Is it a label? */
	return (INT_CONSTANT);
badchar:
	stmtbuf[0] = *nextch++;
	return (UNKNOWN);
}


/*
 * This routine is the top level of the Scanner, called to get a lexeme.
 *
 *	return value - lexeme value
 */
int 
yylex()
{
	static int tokno;
	static int nt;
/*!!! */
        int tkn;
/* Initialize the token value and length. */

	/* yytext[0] = '\0';*/
	yyleng = 0;

	switch (lexstate) {
new_stmt:
 	    case OVER:
                            /*(void)fprintf(stderr,"yylex: OVER");*/
	        return XEOF;
	    case NEWSTMT:	/* need a new statement */
                if(free_form)
                   nt = getcds_ff();
                else
		   nt = getcds();
		if (nt == STEOF) {
		     nt = XEOF;
		     if (pcbfirst != NULL)
			  goto commentscase;
		     else return (XEOF);
		} /* if getcds() == SEOF */

		lastend = stkey == ENDUNIT;

	/*
	 * Now, crunch the new statement.  If nothing's left after crunching,
	 * then go get another statement. 
	 */

		crunch();
		if (nextch > lastch)
			goto new_stmt;
		tokno = 0;
		lexstate = FIRSTTOKEN;
		yystno = stno;
		stno = nxtstno;
		yyleng = 0;
		return (LABEL);

first:
	    case FIRSTTOKEN:	/* first step on a statement */ 
		(void) analyz();
                lexstate = OTHERTOKEN;
		tokno = 1;
/* !!! */
               /* (void)fprintf(stderr,"yylex:token N%d %d line %d\n",tokno,stkey,yylineno); */
		return (stkey);

	    case OTHERTOKEN:	/* return next token */
		if (stkey == FORMAT)
		        goto reteos;
/*		if (stkey == READ)
		        goto reteos;
		if (stkey == WRITE)
		        goto reteos;
*/		if (nextch > lastch) {
			nextch = lastch + 1;
			goto reteos;
		}
 		++tokno;
                
		if ((stkey == LOGICALIF || stkey == ELSEIF || stkey == FORALL)
                      && parlev == 0 &&tokno > 3)
			goto first;
		if (stkey == ASSIGN && tokno == 3 && nextch < lastch &&
		    nextch[0] == 't' && nextch[1] == 'o') {
			nextch += 2;
			return (TO);
		}
		if ((stkey == CONSTRUCT_ID) && (tokno == 2) && (nextch[0] == ':'))
		     lexstate = FIRSTTOKEN;
                
                tkn = gettok();
		/*(void)fprintf(stderr,"yylex:token N%d %d\n",tokno,tkn);*/
                return(tkn);
/*!!!		return (gettok()); */

reteos:
	    case RETEOS:
		if (pcbfirst != NULL && stkey != INCLUDE)
		     lexstate = COMMENTS;
		else lexstate = NEWSTMT;
		return (EOLN);
commentscase:
	   case COMMENTS:
		lexstate = COMMENTSEOS;
		flush_comments();
		return (COMMENT);
	   case COMMENTSEOS:
		if (nt == XEOF)
		     lexstate = OVER;
		else lexstate = NEWSTMT;
		return (EOLN);
	}

	if (nt == EOF)
	     if (pcbfirst != NULL) {
		  flush_comments();
		  return (COMMENT);
	     }
	
        return nt;
	
}

/* Comment buffering code */

static void store_comment()
{
    comment_buf *ncb;

    ncb = (comment_buf *)chkalloc(sizeof(comment_buf));
         /* 14.10.2016 Kolganov, 17.10.16 podd. Don't loss '\n' symbol during truncating */
    if ((int)strlen(tempbuf) > COMMENT_BUF_STORE-1)   
    {
        warn("comment too long, truncated to 160 characters", 15);               
	tempbuf[COMMENT_BUF_STORE-2] = '\n';
        tempbuf[COMMENT_BUF_STORE-1] = '\0';
    }

    ncb->buf[0] = '\0';
    strcpy(ncb->buf, tempbuf);
    /* (void)fprintf(stderr,"store_comment_end_permbuff:tempbuf = %s \n",tempbuf);*/
    ncb->next = NULL;
    if (!cbfirst) {
        cbfirst = ncb;
        cblast = cbfirst;
    }
    else
    {
        cblast->next = ncb;
        cblast = ncb;
    }
}

static void
store_comment_end_permbuff()

{
     comment_buf *ncb;
     
     ncb = (comment_buf *) chkalloc(sizeof(comment_buf));
     ncb->buf[0]='\0';
     strcpy(ncb->buf,tempbuf);
     /* (void)fprintf(stderr,"store_comment_end_permbuff:tempbuf = %s \n",tempbuf);*/
     ncb->next = NULL;
     if (!pcbfirst) {
	  pcbfirst = ncb;
	  pcblast = pcbfirst;
     }
     else 
     {
	  pcblast->next = ncb;
	  pcblast = ncb;
     }
}

static void
move_comments_to_permbuffer()
{
     if (pcbfirst == NULL)
     {
	  pcbfirst = cbfirst;
	  pcblast = cblast;
/*  (void)fprintf(stderr,"move_comments_to_permbuff: first\n");*/
     }
     else 
     {
	  if (cbfirst != NULL) 
	  {
	       pcblast->next = cbfirst;
	       pcblast = cblast;
/*  (void)fprintf(stderr,"move_comments_to_permbuff: next\n");*/
	  }
	  
     }
     
     cbfirst = cblast = NULL;
/*        (void)fprintf(stderr,"move_comments_to_permbuff:\n");*/
}

     
static void
flush_comments()
{
     comment_buf *ncb;
     int len = 0;

     for (ncb = pcbfirst; ncb; ncb = ncb->next)
	  len = len + strlen(ncb->buf) + 1;
     
     commentbuf = (char *) chkalloc(len * sizeof(char));
     
     for (ncb = pcbfirst; ncb; ncb = ncb->next) 
	  strcat(commentbuf, ncb->buf);
     

     pcbfirst = pcblast = NULL;
     /*   (void)fprintf(stderr,"%s\n", commentbuf);*/
/*    printf("%s\n", commentbuf);*/
     
}     





