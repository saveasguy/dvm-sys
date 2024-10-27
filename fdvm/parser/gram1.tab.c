/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     PERCENT = 1,
     AMPERSAND = 2,
     ASTER = 3,
     CLUSTER = 4,
     COLON = 5,
     COMMA = 6,
     DASTER = 7,
     DEFINED_OPERATOR = 8,
     DOT = 9,
     DQUOTE = 10,
     GLOBAL_A = 11,
     LEFTAB = 12,
     LEFTPAR = 13,
     MINUS = 14,
     PLUS = 15,
     POINT_TO = 16,
     QUOTE = 17,
     RIGHTAB = 18,
     RIGHTPAR = 19,
     AND = 20,
     DSLASH = 21,
     EQV = 22,
     EQ = 23,
     EQUAL = 24,
     FFALSE = 25,
     GE = 26,
     GT = 27,
     LE = 28,
     LT = 29,
     NE = 30,
     NEQV = 31,
     NOT = 32,
     OR = 33,
     TTRUE = 34,
     SLASH = 35,
     XOR = 36,
     REFERENCE = 37,
     AT = 38,
     ACROSS = 39,
     ALIGN_WITH = 40,
     ALIGN = 41,
     ALLOCATABLE = 42,
     ALLOCATE = 43,
     ARITHIF = 44,
     ASSIGNMENT = 45,
     ASSIGN = 46,
     ASSIGNGOTO = 47,
     ASYNCHRONOUS = 48,
     ASYNCID = 49,
     ASYNCWAIT = 50,
     BACKSPACE = 51,
     BAD_CCONST = 52,
     BAD_SYMBOL = 53,
     BARRIER = 54,
     BLOCKDATA = 55,
     BLOCK = 56,
     BOZ_CONSTANT = 57,
     BYTE = 58,
     CALL = 59,
     CASE = 60,
     CHARACTER = 61,
     CHAR_CONSTANT = 62,
     CHECK = 63,
     CLOSE = 64,
     COMMON = 65,
     COMPLEX = 66,
     COMPGOTO = 67,
     CONSISTENT_GROUP = 68,
     CONSISTENT_SPEC = 69,
     CONSISTENT_START = 70,
     CONSISTENT_WAIT = 71,
     CONSISTENT = 72,
     CONSTRUCT_ID = 73,
     CONTAINS = 74,
     CONTINUE = 75,
     CORNER = 76,
     CYCLE = 77,
     DATA = 78,
     DEALLOCATE = 79,
     HPF_TEMPLATE = 80,
     DEBUG = 81,
     DEFAULT_CASE = 82,
     DEFINE = 83,
     DERIVED = 84,
     DIMENSION = 85,
     DISTRIBUTE = 86,
     DOWHILE = 87,
     DOUBLEPRECISION = 88,
     DOUBLECOMPLEX = 89,
     DP_CONSTANT = 90,
     DVM_POINTER = 91,
     DYNAMIC = 92,
     ELEMENTAL = 93,
     ELSE = 94,
     ELSEIF = 95,
     ELSEWHERE = 96,
     ENDASYNCHRONOUS = 97,
     ENDDEBUG = 98,
     ENDINTERVAL = 99,
     ENDUNIT = 100,
     ENDDO = 101,
     ENDFILE = 102,
     ENDFORALL = 103,
     ENDIF = 104,
     ENDINTERFACE = 105,
     ENDMODULE = 106,
     ENDON = 107,
     ENDSELECT = 108,
     ENDTASK_REGION = 109,
     ENDTYPE = 110,
     ENDWHERE = 111,
     ENTRY = 112,
     EXIT = 113,
     EOLN = 114,
     EQUIVALENCE = 115,
     ERROR = 116,
     EXTERNAL = 117,
     F90 = 118,
     FIND = 119,
     FORALL = 120,
     FORMAT = 121,
     FUNCTION = 122,
     GATE = 123,
     GEN_BLOCK = 124,
     HEAP = 125,
     HIGH = 126,
     IDENTIFIER = 127,
     IMPLICIT = 128,
     IMPLICITNONE = 129,
     INCLUDE_TO = 130,
     INCLUDE = 131,
     INDEPENDENT = 132,
     INDIRECT_ACCESS = 133,
     INDIRECT_GROUP = 134,
     INDIRECT = 135,
     INHERIT = 136,
     INQUIRE = 137,
     INTERFACEASSIGNMENT = 138,
     INTERFACEOPERATOR = 139,
     INTERFACE = 140,
     INTRINSIC = 141,
     INTEGER = 142,
     INTENT = 143,
     INTERVAL = 144,
     INOUT = 145,
     IN = 146,
     INT_CONSTANT = 147,
     LABEL = 148,
     LABEL_DECLARE = 149,
     LET = 150,
     LOCALIZE = 151,
     LOGICAL = 152,
     LOGICALIF = 153,
     LOOP = 154,
     LOW = 155,
     MAXLOC = 156,
     MAX = 157,
     MAP = 158,
     MINLOC = 159,
     MIN = 160,
     MODULE_PROCEDURE = 161,
     MODULE = 162,
     MULT_BLOCK = 163,
     NAMEEQ = 164,
     NAMELIST = 165,
     NEW_VALUE = 166,
     NEW = 167,
     NULLIFY = 168,
     OCTAL_CONSTANT = 169,
     ONLY = 170,
     ON = 171,
     ON_DIR = 172,
     ONTO = 173,
     OPEN = 174,
     OPERATOR = 175,
     OPTIONAL = 176,
     OTHERWISE = 177,
     OUT = 178,
     OWN = 179,
     PARALLEL = 180,
     PARAMETER = 181,
     PAUSE = 182,
     PLAINDO = 183,
     PLAINGOTO = 184,
     POINTER = 185,
     POINTERLET = 186,
     PREFETCH = 187,
     PRINT = 188,
     PRIVATE = 189,
     PRODUCT = 190,
     PROGRAM = 191,
     PUBLIC = 192,
     PURE = 193,
     RANGE = 194,
     READ = 195,
     REALIGN_WITH = 196,
     REALIGN = 197,
     REAL = 198,
     REAL_CONSTANT = 199,
     RECURSIVE = 200,
     REDISTRIBUTE_NEW = 201,
     REDISTRIBUTE = 202,
     REDUCTION_GROUP = 203,
     REDUCTION_START = 204,
     REDUCTION_WAIT = 205,
     REDUCTION = 206,
     REMOTE_ACCESS_SPEC = 207,
     REMOTE_ACCESS = 208,
     REMOTE_GROUP = 209,
     RESET = 210,
     RESULT = 211,
     RETURN = 212,
     REWIND = 213,
     SAVE = 214,
     SECTION = 215,
     SELECT = 216,
     SEQUENCE = 217,
     SHADOW_ADD = 218,
     SHADOW_COMPUTE = 219,
     SHADOW_GROUP = 220,
     SHADOW_RENEW = 221,
     SHADOW_START_SPEC = 222,
     SHADOW_START = 223,
     SHADOW_WAIT_SPEC = 224,
     SHADOW_WAIT = 225,
     SHADOW = 226,
     STAGE = 227,
     STATIC = 228,
     STAT = 229,
     STOP = 230,
     SUBROUTINE = 231,
     SUM = 232,
     SYNC = 233,
     TARGET = 234,
     TASK = 235,
     TASK_REGION = 236,
     THEN = 237,
     TO = 238,
     TRACEON = 239,
     TRACEOFF = 240,
     TRUNC = 241,
     TYPE = 242,
     TYPE_DECL = 243,
     UNDER = 244,
     UNKNOWN = 245,
     USE = 246,
     VIRTUAL = 247,
     VARIABLE = 248,
     WAIT = 249,
     WHERE = 250,
     WHERE_ASSIGN = 251,
     WHILE = 252,
     WITH = 253,
     WRITE = 254,
     COMMENT = 255,
     WGT_BLOCK = 256,
     HPF_PROCESSORS = 257,
     IOSTAT = 258,
     ERR = 259,
     END = 260,
     OMPDVM_ATOMIC = 261,
     OMPDVM_BARRIER = 262,
     OMPDVM_COPYIN = 263,
     OMPDVM_COPYPRIVATE = 264,
     OMPDVM_CRITICAL = 265,
     OMPDVM_ONETHREAD = 266,
     OMPDVM_DO = 267,
     OMPDVM_DYNAMIC = 268,
     OMPDVM_ENDCRITICAL = 269,
     OMPDVM_ENDDO = 270,
     OMPDVM_ENDMASTER = 271,
     OMPDVM_ENDORDERED = 272,
     OMPDVM_ENDPARALLEL = 273,
     OMPDVM_ENDPARALLELDO = 274,
     OMPDVM_ENDPARALLELSECTIONS = 275,
     OMPDVM_ENDPARALLELWORKSHARE = 276,
     OMPDVM_ENDSECTIONS = 277,
     OMPDVM_ENDSINGLE = 278,
     OMPDVM_ENDWORKSHARE = 279,
     OMPDVM_FIRSTPRIVATE = 280,
     OMPDVM_FLUSH = 281,
     OMPDVM_GUIDED = 282,
     OMPDVM_LASTPRIVATE = 283,
     OMPDVM_MASTER = 284,
     OMPDVM_NOWAIT = 285,
     OMPDVM_NONE = 286,
     OMPDVM_NUM_THREADS = 287,
     OMPDVM_ORDERED = 288,
     OMPDVM_PARALLEL = 289,
     OMPDVM_PARALLELDO = 290,
     OMPDVM_PARALLELSECTIONS = 291,
     OMPDVM_PARALLELWORKSHARE = 292,
     OMPDVM_RUNTIME = 293,
     OMPDVM_SECTION = 294,
     OMPDVM_SECTIONS = 295,
     OMPDVM_SCHEDULE = 296,
     OMPDVM_SHARED = 297,
     OMPDVM_SINGLE = 298,
     OMPDVM_THREADPRIVATE = 299,
     OMPDVM_WORKSHARE = 300,
     OMPDVM_NODES = 301,
     OMPDVM_IF = 302,
     IAND = 303,
     IEOR = 304,
     IOR = 305,
     ACC_REGION = 306,
     ACC_END_REGION = 307,
     ACC_CHECKSECTION = 308,
     ACC_END_CHECKSECTION = 309,
     ACC_GET_ACTUAL = 310,
     ACC_ACTUAL = 311,
     ACC_TARGETS = 312,
     ACC_ASYNC = 313,
     ACC_HOST = 314,
     ACC_CUDA = 315,
     ACC_LOCAL = 316,
     ACC_INLOCAL = 317,
     ACC_CUDA_BLOCK = 318,
     ACC_ROUTINE = 319,
     ACC_TIE = 320,
     BY = 321,
     IO_MODE = 322,
     CP_CREATE = 323,
     CP_LOAD = 324,
     CP_SAVE = 325,
     CP_WAIT = 326,
     FILES = 327,
     VARLIST = 328,
     STATUS = 329,
     EXITINTERVAL = 330,
     TEMPLATE_CREATE = 331,
     TEMPLATE_DELETE = 332,
     SPF_ANALYSIS = 333,
     SPF_PARALLEL = 334,
     SPF_TRANSFORM = 335,
     SPF_NOINLINE = 336,
     SPF_PARALLEL_REG = 337,
     SPF_END_PARALLEL_REG = 338,
     SPF_EXPAND = 339,
     SPF_FISSION = 340,
     SPF_SHRINK = 341,
     SPF_CHECKPOINT = 342,
     SPF_EXCEPT = 343,
     SPF_FILES_COUNT = 344,
     SPF_INTERVAL = 345,
     SPF_TIME = 346,
     SPF_ITER = 347,
     SPF_FLEXIBLE = 348,
     SPF_APPLY_REGION = 349,
     SPF_APPLY_FRAGMENT = 350,
     SPF_CODE_COVERAGE = 351,
     SPF_UNROLL = 352,
     SPF_MERGE = 353,
     SPF_COVER = 354,
     SPF_PROCESS_PRIVATE = 355,
     BINARY_OP = 358,
     UNARY_OP = 359
   };
#endif
/* Tokens.  */
#define PERCENT 1
#define AMPERSAND 2
#define ASTER 3
#define CLUSTER 4
#define COLON 5
#define COMMA 6
#define DASTER 7
#define DEFINED_OPERATOR 8
#define DOT 9
#define DQUOTE 10
#define GLOBAL_A 11
#define LEFTAB 12
#define LEFTPAR 13
#define MINUS 14
#define PLUS 15
#define POINT_TO 16
#define QUOTE 17
#define RIGHTAB 18
#define RIGHTPAR 19
#define AND 20
#define DSLASH 21
#define EQV 22
#define EQ 23
#define EQUAL 24
#define FFALSE 25
#define GE 26
#define GT 27
#define LE 28
#define LT 29
#define NE 30
#define NEQV 31
#define NOT 32
#define OR 33
#define TTRUE 34
#define SLASH 35
#define XOR 36
#define REFERENCE 37
#define AT 38
#define ACROSS 39
#define ALIGN_WITH 40
#define ALIGN 41
#define ALLOCATABLE 42
#define ALLOCATE 43
#define ARITHIF 44
#define ASSIGNMENT 45
#define ASSIGN 46
#define ASSIGNGOTO 47
#define ASYNCHRONOUS 48
#define ASYNCID 49
#define ASYNCWAIT 50
#define BACKSPACE 51
#define BAD_CCONST 52
#define BAD_SYMBOL 53
#define BARRIER 54
#define BLOCKDATA 55
#define BLOCK 56
#define BOZ_CONSTANT 57
#define BYTE 58
#define CALL 59
#define CASE 60
#define CHARACTER 61
#define CHAR_CONSTANT 62
#define CHECK 63
#define CLOSE 64
#define COMMON 65
#define COMPLEX 66
#define COMPGOTO 67
#define CONSISTENT_GROUP 68
#define CONSISTENT_SPEC 69
#define CONSISTENT_START 70
#define CONSISTENT_WAIT 71
#define CONSISTENT 72
#define CONSTRUCT_ID 73
#define CONTAINS 74
#define CONTINUE 75
#define CORNER 76
#define CYCLE 77
#define DATA 78
#define DEALLOCATE 79
#define HPF_TEMPLATE 80
#define DEBUG 81
#define DEFAULT_CASE 82
#define DEFINE 83
#define DERIVED 84
#define DIMENSION 85
#define DISTRIBUTE 86
#define DOWHILE 87
#define DOUBLEPRECISION 88
#define DOUBLECOMPLEX 89
#define DP_CONSTANT 90
#define DVM_POINTER 91
#define DYNAMIC 92
#define ELEMENTAL 93
#define ELSE 94
#define ELSEIF 95
#define ELSEWHERE 96
#define ENDASYNCHRONOUS 97
#define ENDDEBUG 98
#define ENDINTERVAL 99
#define ENDUNIT 100
#define ENDDO 101
#define ENDFILE 102
#define ENDFORALL 103
#define ENDIF 104
#define ENDINTERFACE 105
#define ENDMODULE 106
#define ENDON 107
#define ENDSELECT 108
#define ENDTASK_REGION 109
#define ENDTYPE 110
#define ENDWHERE 111
#define ENTRY 112
#define EXIT 113
#define EOLN 114
#define EQUIVALENCE 115
#define ERROR 116
#define EXTERNAL 117
#define F90 118
#define FIND 119
#define FORALL 120
#define FORMAT 121
#define FUNCTION 122
#define GATE 123
#define GEN_BLOCK 124
#define HEAP 125
#define HIGH 126
#define IDENTIFIER 127
#define IMPLICIT 128
#define IMPLICITNONE 129
#define INCLUDE_TO 130
#define INCLUDE 131
#define INDEPENDENT 132
#define INDIRECT_ACCESS 133
#define INDIRECT_GROUP 134
#define INDIRECT 135
#define INHERIT 136
#define INQUIRE 137
#define INTERFACEASSIGNMENT 138
#define INTERFACEOPERATOR 139
#define INTERFACE 140
#define INTRINSIC 141
#define INTEGER 142
#define INTENT 143
#define INTERVAL 144
#define INOUT 145
#define IN 146
#define INT_CONSTANT 147
#define LABEL 148
#define LABEL_DECLARE 149
#define LET 150
#define LOCALIZE 151
#define LOGICAL 152
#define LOGICALIF 153
#define LOOP 154
#define LOW 155
#define MAXLOC 156
#define MAX 157
#define MAP 158
#define MINLOC 159
#define MIN 160
#define MODULE_PROCEDURE 161
#define MODULE 162
#define MULT_BLOCK 163
#define NAMEEQ 164
#define NAMELIST 165
#define NEW_VALUE 166
#define NEW 167
#define NULLIFY 168
#define OCTAL_CONSTANT 169
#define ONLY 170
#define ON 171
#define ON_DIR 172
#define ONTO 173
#define OPEN 174
#define OPERATOR 175
#define OPTIONAL 176
#define OTHERWISE 177
#define OUT 178
#define OWN 179
#define PARALLEL 180
#define PARAMETER 181
#define PAUSE 182
#define PLAINDO 183
#define PLAINGOTO 184
#define POINTER 185
#define POINTERLET 186
#define PREFETCH 187
#define PRINT 188
#define PRIVATE 189
#define PRODUCT 190
#define PROGRAM 191
#define PUBLIC 192
#define PURE 193
#define RANGE 194
#define READ 195
#define REALIGN_WITH 196
#define REALIGN 197
#define REAL 198
#define REAL_CONSTANT 199
#define RECURSIVE 200
#define REDISTRIBUTE_NEW 201
#define REDISTRIBUTE 202
#define REDUCTION_GROUP 203
#define REDUCTION_START 204
#define REDUCTION_WAIT 205
#define REDUCTION 206
#define REMOTE_ACCESS_SPEC 207
#define REMOTE_ACCESS 208
#define REMOTE_GROUP 209
#define RESET 210
#define RESULT 211
#define RETURN 212
#define REWIND 213
#define SAVE 214
#define SECTION 215
#define SELECT 216
#define SEQUENCE 217
#define SHADOW_ADD 218
#define SHADOW_COMPUTE 219
#define SHADOW_GROUP 220
#define SHADOW_RENEW 221
#define SHADOW_START_SPEC 222
#define SHADOW_START 223
#define SHADOW_WAIT_SPEC 224
#define SHADOW_WAIT 225
#define SHADOW 226
#define STAGE 227
#define STATIC 228
#define STAT 229
#define STOP 230
#define SUBROUTINE 231
#define SUM 232
#define SYNC 233
#define TARGET 234
#define TASK 235
#define TASK_REGION 236
#define THEN 237
#define TO 238
#define TRACEON 239
#define TRACEOFF 240
#define TRUNC 241
#define TYPE 242
#define TYPE_DECL 243
#define UNDER 244
#define UNKNOWN 245
#define USE 246
#define VIRTUAL 247
#define VARIABLE 248
#define WAIT 249
#define WHERE 250
#define WHERE_ASSIGN 251
#define WHILE 252
#define WITH 253
#define WRITE 254
#define COMMENT 255
#define WGT_BLOCK 256
#define HPF_PROCESSORS 257
#define IOSTAT 258
#define ERR 259
#define END 260
#define OMPDVM_ATOMIC 261
#define OMPDVM_BARRIER 262
#define OMPDVM_COPYIN 263
#define OMPDVM_COPYPRIVATE 264
#define OMPDVM_CRITICAL 265
#define OMPDVM_ONETHREAD 266
#define OMPDVM_DO 267
#define OMPDVM_DYNAMIC 268
#define OMPDVM_ENDCRITICAL 269
#define OMPDVM_ENDDO 270
#define OMPDVM_ENDMASTER 271
#define OMPDVM_ENDORDERED 272
#define OMPDVM_ENDPARALLEL 273
#define OMPDVM_ENDPARALLELDO 274
#define OMPDVM_ENDPARALLELSECTIONS 275
#define OMPDVM_ENDPARALLELWORKSHARE 276
#define OMPDVM_ENDSECTIONS 277
#define OMPDVM_ENDSINGLE 278
#define OMPDVM_ENDWORKSHARE 279
#define OMPDVM_FIRSTPRIVATE 280
#define OMPDVM_FLUSH 281
#define OMPDVM_GUIDED 282
#define OMPDVM_LASTPRIVATE 283
#define OMPDVM_MASTER 284
#define OMPDVM_NOWAIT 285
#define OMPDVM_NONE 286
#define OMPDVM_NUM_THREADS 287
#define OMPDVM_ORDERED 288
#define OMPDVM_PARALLEL 289
#define OMPDVM_PARALLELDO 290
#define OMPDVM_PARALLELSECTIONS 291
#define OMPDVM_PARALLELWORKSHARE 292
#define OMPDVM_RUNTIME 293
#define OMPDVM_SECTION 294
#define OMPDVM_SECTIONS 295
#define OMPDVM_SCHEDULE 296
#define OMPDVM_SHARED 297
#define OMPDVM_SINGLE 298
#define OMPDVM_THREADPRIVATE 299
#define OMPDVM_WORKSHARE 300
#define OMPDVM_NODES 301
#define OMPDVM_IF 302
#define IAND 303
#define IEOR 304
#define IOR 305
#define ACC_REGION 306
#define ACC_END_REGION 307
#define ACC_CHECKSECTION 308
#define ACC_END_CHECKSECTION 309
#define ACC_GET_ACTUAL 310
#define ACC_ACTUAL 311
#define ACC_TARGETS 312
#define ACC_ASYNC 313
#define ACC_HOST 314
#define ACC_CUDA 315
#define ACC_LOCAL 316
#define ACC_INLOCAL 317
#define ACC_CUDA_BLOCK 318
#define ACC_ROUTINE 319
#define ACC_TIE 320
#define BY 321
#define IO_MODE 322
#define CP_CREATE 323
#define CP_LOAD 324
#define CP_SAVE 325
#define CP_WAIT 326
#define FILES 327
#define VARLIST 328
#define STATUS 329
#define EXITINTERVAL 330
#define TEMPLATE_CREATE 331
#define TEMPLATE_DELETE 332
#define SPF_ANALYSIS 333
#define SPF_PARALLEL 334
#define SPF_TRANSFORM 335
#define SPF_NOINLINE 336
#define SPF_PARALLEL_REG 337
#define SPF_END_PARALLEL_REG 338
#define SPF_EXPAND 339
#define SPF_FISSION 340
#define SPF_SHRINK 341
#define SPF_CHECKPOINT 342
#define SPF_EXCEPT 343
#define SPF_FILES_COUNT 344
#define SPF_INTERVAL 345
#define SPF_TIME 346
#define SPF_ITER 347
#define SPF_FLEXIBLE 348
#define SPF_APPLY_REGION 349
#define SPF_APPLY_FRAGMENT 350
#define SPF_CODE_COVERAGE 351
#define SPF_UNROLL 352
#define SPF_MERGE 353
#define SPF_COVER 354
#define SPF_PROCESS_PRIVATE 355
#define BINARY_OP 358
#define UNARY_OP 359




/* Copy the first part of user declarations.  */
#line 357 "gram1.y"

#include <string.h>
#include "inc.h"
#include "extern.h"
#include "defines.h"
#include "fdvm.h"
#include "fm.h"

/* We may use builtin alloca */
#include "compatible.h"
#ifdef _NEEDALLOCAH_
#  include <alloca.h>
#endif

#define EXTEND_NODE 2  /* move the definition to h/ files. */

extern PTR_BFND global_bfnd, pred_bfnd;
extern PTR_SYMB star_symb;
extern PTR_SYMB global_list;
extern PTR_TYPE global_bool;
extern PTR_TYPE global_int;
extern PTR_TYPE global_float;
extern PTR_TYPE global_double;
extern PTR_TYPE global_char;
extern PTR_TYPE global_string;
extern PTR_TYPE global_string_2;
extern PTR_TYPE global_complex;
extern PTR_TYPE global_dcomplex;
extern PTR_TYPE global_gate;
extern PTR_TYPE global_event;
extern PTR_TYPE global_sequence;
extern PTR_TYPE global_default;
extern PTR_LABEL thislabel;
extern PTR_CMNT comments, cur_comment;
extern PTR_BFND last_bfnd;
extern PTR_TYPE impltype[];
extern int nioctl;
extern int maxdim;
extern long yystno;	/* statement label */
extern char stmtbuf[];	/* input buffer */
extern char *commentbuf;	/* comments buffer from scanner */
extern PTR_BLOB head_blob;
extern PTR_BLOB cur_blob;
extern PTR_TYPE vartype; /* variable type */
extern int end_group;
extern char saveall;
extern int privateall;
extern int needkwd;
extern int implkwd;
extern int opt_kwd_hedr;
/* added for FORTRAN 90 */
extern PTR_LLND first_unresolved_call;
extern PTR_LLND last_unresolved_call;
extern int data_stat;
extern char yyquote;

extern int warn_all;
extern int statement_kind; /* kind of statement: 1 - HPF-DVM-directive, 0 - Fortran statement*/ 
int extend_flag = 0;

static int do_name_err;
static int ndim;	/* number of dimension */
/*!!! hpf */
static int explicit_shape; /*  1 if shape specification is explicit */
/* static int varleng;*/	/* variable size */
static int lastwasbranch = NO;	/* set if last stmt was a branch stmt */
static int thiswasbranch = NO;	/* set if this stmt is a branch stmt */
static PTR_SYMB type_var = SMNULL;
static PTR_LLND stat_alloc = LLNULL; /* set if ALLOCATE/DEALLOCATE stmt has STAT-clause*/
/* static int subscripts_status = 0; */
static int type_options,type_opt;   /* The various options used to declare a name -
                                      RECURSIVE, POINTER, OPTIONAL etc.         */
static PTR_BFND module_scope;
static int position = IN_OUTSIDE;            
static int attr_ndim;           /* number of dimensions in DIMENSION (array_spec)
                                   attribute declaration */
static PTR_LLND attr_dims;     /* low level representation of array_spec in
                                   DIMENSION (array_spec) attribute declarartion. */
static int in_vec = NO;	      /* set if processing array constructor */


/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 438 "gram1.y"
{
    int token;
    char charv;
    char *charp;
    PTR_BFND bf_node;
    PTR_LLND ll_node;
    PTR_SYMB symbol;
    PTR_TYPE data_type;
    PTR_HASH hash_entry;
    PTR_LABEL label;
}
/* Line 187 of yacc.c.  */
#line 907 "gram1.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */
#line 646 "gram1.y"

void add_scope_level();
void delete_beyond_scope_level();
PTR_HASH look_up_sym();
PTR_HASH just_look_up_sym();
PTR_HASH just_look_up_sym_in_scope();
PTR_HASH look_up_op();
PTR_SYMB make_constant();
PTR_SYMB make_scalar();
PTR_SYMB make_array();
PTR_SYMB make_pointer();
PTR_SYMB make_function();
PTR_SYMB make_external();
PTR_SYMB make_intrinsic();
PTR_SYMB make_procedure();
PTR_SYMB make_process();
PTR_SYMB make_program();
PTR_SYMB make_module();
PTR_SYMB make_common();
PTR_SYMB make_parallel_region();
PTR_SYMB make_derived_type();
PTR_SYMB make_local_entity();
PTR_SYMB make_global_entity();
PTR_TYPE make_type_node();
PTR_TYPE lookup_type(), make_type();
void     process_type();
void     process_interface();
void     bind();
void     late_bind_if_needed();
PTR_SYMB component();
PTR_SYMB lookup_type_symbol();
PTR_SYMB resolve_overloading();
PTR_BFND cur_scope();
PTR_BFND subroutine_call();
PTR_BFND process_call();
PTR_LLND deal_with_options();
PTR_LLND intrinsic_op_node();
PTR_LLND defined_op_node();
int is_substring_ref();
int is_array_section_ref();
PTR_LLND dim_expr(); 
PTR_BFND exit_stat();
PTR_BFND make_do();
PTR_BFND make_pardo();
PTR_BFND make_enddoall();
PTR_TYPE install_array(); 
PTR_SYMB install_entry(); 
void install_param_list();
PTR_LLND construct_entry_list();
void copy_sym_data();
PTR_LLND check_and_install();
PTR_HASH look_up();
PTR_BFND get_bfnd(); 
PTR_BLOB make_blob();
PTR_LABEL make_label();
PTR_LABEL make_label_node();
int is_interface_stat();
PTR_LLND make_llnd (); 
PTR_LLND make_llnd_label (); 
PTR_TYPE make_sa_type(); 
PTR_SYMB procedure_call();
PTR_BFND proc_list();
PTR_SYMB set_id_list();
PTR_LLND set_ll_list();
PTR_LLND add_to_lowLevelList(), add_to_lowList();
PTR_BFND set_stat_list() ;
PTR_BLOB follow_blob();
PTR_SYMB proc_decl_init();
PTR_CMNT make_comment();
PTR_HASH correct_symtab();
char *copyn();
char *convic();
char *StringConcatenation();
int atoi();
PTR_BFND make_logif();
PTR_BFND make_if();
PTR_BFND make_forall();
void startproc();
void match_parameters();
void make_else();
void make_elseif();
void make_endif();
void make_elsewhere();
void make_elsewhere_mask();
void make_endwhere();
void make_endforall();
void make_endselect();
void make_extend();
void make_endextend();
void make_section();
void make_section_extend();
void doinclude();
void endproc();
void err();
void execerr();
void flline();
void warn();
void warn1();
void newprog();
void set_type();
void dclerr();
void enddcl();
void install_const();
void setimpl();
void copy_module_scope();
void delete_symbol();
void replace_symbol_in_expr();
long convci();
void set_expr_type();
void errstr();
void yyerror();
void set_blobs();
void make_loop();
void startioctl();
void endioctl();
void redefine_func_arg_type();
int isResultVar();
int yylex();

/* used by FORTRAN M */
PTR_BFND make_processdo();
PTR_BFND make_processes();
PTR_BFND make_endprocesses();

PTR_BFND make_endparallel();/*OMP*/
PTR_BFND make_parallel();/*OMP*/
PTR_BFND make_endsingle();/*OMP*/
PTR_BFND make_single();/*OMP*/
PTR_BFND make_endmaster();/*OMP*/
PTR_BFND make_master();/*OMP*/
PTR_BFND make_endordered();/*OMP*/
PTR_BFND make_ordered();/*OMP*/
PTR_BFND make_endcritical();/*OMP*/
PTR_BFND make_critical();/*OMP*/
PTR_BFND make_endsections();/*OMP*/
PTR_BFND make_sections();/*OMP*/
PTR_BFND make_ompsection();/*OMP*/
PTR_BFND make_endparallelsections();/*OMP*/
PTR_BFND make_parallelsections();/*OMP*/
PTR_BFND make_endworkshare();/*OMP*/
PTR_BFND make_workshare();/*OMP*/
PTR_BFND make_endparallelworkshare();/*OMP*/
PTR_BFND make_parallelworkshare();/*OMP*/



/* Line 216 of yacc.c.  */
#line 1065 "gram1.tab.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   5895

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  360
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  541
/* YYNRULES -- Number of rules.  */
#define YYNRULES  1302
/* YYNRULES -- Number of states.  */
#define YYNSTATES  2596

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   359

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint16 yytranslate[] =
{
       0,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   110,   111,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,   139,   140,   141,
     142,   143,   144,   145,   146,   147,   148,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
     162,   163,   164,   165,   166,   167,   168,   169,   170,   171,
     172,   173,   174,   175,   176,   177,   178,   179,   180,   181,
     182,   183,   184,   185,   186,   187,   188,   189,   190,   191,
     192,   193,   194,   195,   196,   197,   198,   199,   200,   201,
     202,   203,   204,   205,   206,   207,   208,   209,   210,   211,
     212,   213,   214,   215,   216,   217,   218,   219,   220,   221,
     222,   223,   224,   225,   226,   227,   228,   229,   230,   231,
     232,   233,   234,   235,   236,   237,   238,   239,   240,   241,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   284,   285,   286,   287,   288,   289,   290,   291,
     292,   293,   294,   295,   296,   297,   298,   299,   300,   301,
     302,   303,   304,   305,   306,   307,   308,   309,   310,   311,
     312,   313,   314,   315,   316,   317,   318,   319,   320,   321,
     322,   323,   324,   325,   326,   327,   328,   329,   330,   331,
     332,   333,   334,   335,   336,   337,   338,   339,   340,   341,
     342,   343,   344,   345,   346,   347,   348,   349,   350,   351,
     352,   353,   354,   355,   356,   357,     1,     2,   358,   359
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     8,    12,    16,    20,    24,    27,
      29,    31,    33,    37,    41,    46,    52,    58,    62,    67,
      71,    72,    75,    78,    81,    83,    85,    90,    96,   101,
     107,   110,   116,   118,   119,   121,   122,   124,   125,   128,
     132,   134,   138,   140,   142,   144,   145,   146,   147,   149,
     151,   153,   155,   157,   159,   161,   163,   165,   167,   169,
     171,   173,   176,   181,   184,   190,   192,   194,   196,   198,
     200,   202,   204,   206,   208,   210,   212,   214,   217,   221,
     227,   233,   236,   238,   240,   242,   244,   246,   248,   250,
     252,   254,   256,   258,   260,   262,   264,   266,   268,   270,
     272,   274,   276,   278,   283,   291,   294,   298,   306,   313,
     314,   317,   323,   325,   330,   332,   334,   336,   339,   341,
     346,   348,   350,   352,   354,   356,   358,   361,   364,   367,
     369,   371,   379,   383,   388,   392,   397,   401,   404,   410,
     411,   414,   417,   423,   424,   429,   435,   436,   439,   443,
     445,   447,   449,   451,   453,   455,   457,   459,   461,   463,
     464,   466,   472,   479,   486,   487,   489,   495,   505,   507,
     509,   512,   515,   516,   517,   520,   523,   529,   534,   539,
     543,   548,   552,   557,   561,   565,   570,   576,   580,   585,
     591,   595,   599,   601,   605,   608,   613,   617,   622,   626,
     630,   634,   638,   642,   646,   648,   653,   655,   657,   662,
     666,   667,   668,   673,   675,   679,   681,   685,   688,   692,
     696,   701,   704,   705,   707,   709,   713,   719,   721,   725,
     726,   728,   736,   738,   742,   745,   748,   752,   754,   756,
     761,   765,   768,   770,   772,   774,   776,   780,   782,   786,
     788,   790,   797,   799,   801,   804,   807,   809,   813,   815,
     818,   821,   823,   827,   829,   833,   839,   841,   843,   845,
     848,   851,   855,   859,   861,   865,   869,   871,   875,   877,
     879,   883,   885,   889,   891,   893,   897,   903,   904,   905,
     907,   912,   917,   919,   923,   927,   930,   932,   936,   940,
     947,   954,   962,   964,   966,   970,   972,   974,   976,   980,
     984,   985,   989,   990,   993,   997,   999,  1001,  1004,  1008,
    1010,  1012,  1014,  1018,  1020,  1024,  1026,  1028,  1032,  1037,
    1038,  1041,  1044,  1046,  1048,  1052,  1054,  1058,  1060,  1061,
    1062,  1063,  1066,  1067,  1069,  1071,  1073,  1076,  1079,  1084,
    1086,  1090,  1092,  1096,  1098,  1100,  1102,  1104,  1108,  1112,
    1116,  1120,  1124,  1127,  1130,  1133,  1137,  1141,  1145,  1149,
    1153,  1157,  1161,  1165,  1169,  1173,  1177,  1180,  1184,  1188,
    1190,  1192,  1194,  1196,  1198,  1200,  1206,  1213,  1218,  1224,
    1228,  1230,  1232,  1238,  1243,  1246,  1247,  1249,  1255,  1256,
    1258,  1260,  1264,  1266,  1270,  1273,  1275,  1277,  1279,  1281,
    1283,  1285,  1289,  1293,  1299,  1301,  1303,  1307,  1310,  1316,
    1321,  1326,  1330,  1333,  1335,  1336,  1337,  1344,  1346,  1348,
    1350,  1355,  1361,  1363,  1368,  1374,  1375,  1377,  1381,  1383,
    1385,  1387,  1390,  1394,  1398,  1401,  1403,  1406,  1409,  1412,
    1416,  1424,  1428,  1432,  1434,  1437,  1440,  1442,  1445,  1449,
    1451,  1453,  1455,  1461,  1469,  1470,  1477,  1482,  1494,  1508,
    1513,  1517,  1521,  1529,  1538,  1542,  1544,  1547,  1550,  1554,
    1556,  1560,  1561,  1563,  1564,  1566,  1568,  1571,  1577,  1584,
    1586,  1590,  1594,  1595,  1598,  1600,  1606,  1614,  1615,  1617,
    1621,  1625,  1632,  1638,  1645,  1650,  1656,  1662,  1665,  1667,
    1669,  1680,  1682,  1686,  1691,  1695,  1699,  1703,  1707,  1714,
    1721,  1727,  1736,  1739,  1743,  1747,  1755,  1763,  1764,  1766,
    1771,  1774,  1779,  1781,  1784,  1787,  1789,  1791,  1792,  1793,
    1794,  1797,  1800,  1803,  1806,  1809,  1811,  1814,  1817,  1821,
    1826,  1829,  1833,  1835,  1839,  1843,  1845,  1847,  1849,  1853,
    1855,  1857,  1862,  1868,  1870,  1872,  1876,  1880,  1882,  1887,
    1889,  1891,  1893,  1896,  1899,  1902,  1904,  1908,  1912,  1917,
    1922,  1924,  1928,  1930,  1936,  1938,  1940,  1942,  1946,  1950,
    1954,  1958,  1962,  1966,  1968,  1972,  1978,  1984,  1990,  1991,
    1992,  1994,  1998,  2000,  2002,  2006,  2010,  2014,  2018,  2021,
    2025,  2029,  2030,  2032,  2034,  2036,  2038,  2040,  2042,  2044,
    2046,  2048,  2050,  2052,  2054,  2056,  2058,  2060,  2062,  2064,
    2066,  2068,  2070,  2072,  2074,  2076,  2078,  2080,  2082,  2084,
    2086,  2088,  2090,  2092,  2094,  2096,  2098,  2100,  2102,  2104,
    2106,  2108,  2110,  2112,  2114,  2116,  2118,  2120,  2122,  2124,
    2126,  2128,  2130,  2132,  2134,  2136,  2138,  2140,  2142,  2144,
    2146,  2148,  2150,  2152,  2154,  2156,  2158,  2162,  2166,  2169,
    2173,  2175,  2179,  2181,  2185,  2187,  2191,  2193,  2198,  2202,
    2204,  2208,  2210,  2214,  2219,  2221,  2226,  2231,  2236,  2240,
    2244,  2246,  2250,  2254,  2256,  2260,  2264,  2266,  2270,  2274,
    2276,  2280,  2281,  2287,  2294,  2303,  2305,  2309,  2311,  2313,
    2315,  2320,  2322,  2323,  2326,  2330,  2333,  2338,  2339,  2341,
    2347,  2352,  2359,  2364,  2366,  2371,  2376,  2378,  2385,  2387,
    2391,  2393,  2397,  2399,  2404,  2406,  2408,  2412,  2414,  2416,
    2420,  2422,  2423,  2425,  2428,  2432,  2434,  2437,  2443,  2448,
    2453,  2460,  2462,  2466,  2468,  2470,  2477,  2482,  2484,  2488,
    2490,  2492,  2494,  2496,  2498,  2502,  2504,  2506,  2508,  2515,
    2520,  2522,  2527,  2529,  2531,  2533,  2535,  2540,  2543,  2551,
    2553,  2558,  2560,  2562,  2574,  2575,  2578,  2582,  2584,  2588,
    2590,  2594,  2596,  2600,  2602,  2606,  2608,  2612,  2614,  2618,
    2627,  2629,  2633,  2636,  2639,  2647,  2649,  2653,  2657,  2659,
    2664,  2666,  2670,  2672,  2674,  2675,  2677,  2679,  2682,  2684,
    2686,  2688,  2690,  2692,  2694,  2696,  2698,  2700,  2702,  2704,
    2713,  2720,  2729,  2736,  2738,  2745,  2752,  2759,  2766,  2768,
    2772,  2778,  2780,  2784,  2791,  2793,  2797,  2806,  2813,  2820,
    2825,  2831,  2837,  2838,  2841,  2844,  2845,  2847,  2851,  2853,
    2858,  2866,  2868,  2872,  2876,  2878,  2882,  2888,  2892,  2896,
    2898,  2902,  2904,  2906,  2910,  2914,  2918,  2922,  2933,  2942,
    2953,  2954,  2955,  2957,  2960,  2965,  2970,  2977,  2979,  2981,
    2983,  2985,  2987,  2989,  2991,  2993,  2995,  2997,  2999,  3006,
    3011,  3016,  3020,  3030,  3032,  3034,  3038,  3040,  3046,  3052,
    3062,  3063,  3065,  3067,  3071,  3075,  3079,  3083,  3087,  3094,
    3098,  3102,  3106,  3110,  3118,  3124,  3126,  3128,  3132,  3137,
    3139,  3141,  3145,  3147,  3149,  3153,  3157,  3160,  3164,  3169,
    3174,  3180,  3186,  3188,  3191,  3196,  3201,  3206,  3207,  3209,
    3212,  3220,  3227,  3231,  3235,  3243,  3249,  3251,  3255,  3257,
    3262,  3265,  3269,  3273,  3278,  3285,  3289,  3292,  3296,  3298,
    3300,  3305,  3311,  3315,  3322,  3325,  3330,  3333,  3335,  3339,
    3343,  3344,  3346,  3350,  3353,  3356,  3359,  3362,  3372,  3378,
    3380,  3384,  3387,  3390,  3393,  3403,  3408,  3410,  3414,  3416,
    3418,  3421,  3422,  3430,  3432,  3437,  3439,  3443,  3445,  3447,
    3449,  3466,  3467,  3471,  3475,  3479,  3483,  3490,  3500,  3506,
    3508,  3512,  3518,  3520,  3522,  3524,  3526,  3528,  3530,  3532,
    3534,  3536,  3538,  3540,  3542,  3544,  3546,  3548,  3550,  3552,
    3554,  3556,  3558,  3560,  3562,  3564,  3566,  3568,  3570,  3572,
    3575,  3578,  3583,  3587,  3592,  3598,  3600,  3602,  3604,  3606,
    3608,  3610,  3612,  3614,  3616,  3622,  3625,  3628,  3631,  3634,
    3637,  3643,  3645,  3647,  3649,  3654,  3659,  3664,  3669,  3671,
    3673,  3675,  3677,  3679,  3681,  3683,  3685,  3687,  3689,  3691,
    3693,  3695,  3697,  3699,  3704,  3708,  3713,  3719,  3721,  3723,
    3725,  3727,  3732,  3736,  3739,  3744,  3748,  3753,  3757,  3762,
    3768,  3770,  3772,  3774,  3776,  3778,  3780,  3782,  3790,  3796,
    3798,  3800,  3802,  3804,  3809,  3813,  3818,  3824,  3826,  3828,
    3833,  3837,  3842,  3848,  3850,  3852,  3855,  3857,  3860,  3865,
    3869,  3874,  3878,  3883,  3889,  3891,  3893,  3895,  3897,  3899,
    3901,  3903,  3905,  3907,  3909,  3911,  3914,  3919,  3923,  3926,
    3931,  3935,  3938,  3942,  3945,  3948,  3951,  3954,  3957,  3960,
    3964,  3967,  3973,  3976,  3982,  3985,  3991,  3993,  3995,  3999,
    4003,  4004,  4005,  4007,  4009,  4011,  4013,  4015,  4017,  4019,
    4023,  4026,  4032,  4037,  4040,  4046,  4051,  4054,  4057,  4059,
    4061,  4065,  4068,  4071,  4074,  4079,  4084,  4089,  4094,  4099,
    4104,  4106,  4108,  4110,  4114,  4117,  4120,  4122,  4124,  4128,
    4131,  4134,  4136,  4138,  4140,  4142,  4144,  4146,  4152,  4158,
    4164,  4168,  4179,  4190,  4192,  4196,  4199,  4200,  4207,  4208,
    4215,  4218,  4220,  4224,  4226,  4228,  4230,  4232,  4234,  4240,
    4246,  4252,  4258,  4264,  4266,  4270,  4274,  4276,  4280,  4282,
    4284,  4286,  4292,  4298,  4304,  4306,  4310,  4313,  4319,  4322,
    4328,  4334,  4337,  4343,  4346,  4352,  4354,  4356,  4360,  4366,
    4368,  4372,  4378,  4384,  4390,  4396,  4404,  4406,  4410,  4413,
    4416,  4419,  4422
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     361,     0,    -1,    -1,   361,   362,   116,    -1,   363,   364,
     581,    -1,   363,   381,   581,    -1,   363,   526,   581,    -1,
     363,   133,   377,    -1,   363,   247,    -1,   257,    -1,     1,
      -1,   150,    -1,   193,   365,   372,    -1,    57,   365,   373,
      -1,   233,   365,   367,   374,    -1,   366,   233,   365,   367,
     374,    -1,   124,   365,   368,   374,   370,    -1,   369,   374,
     370,    -1,   114,   371,   374,   370,    -1,   164,   365,   371,
      -1,    -1,   202,   378,    -1,   195,   378,    -1,    95,   378,
      -1,   371,    -1,   371,    -1,   402,   124,   365,   371,    -1,
     402,   366,   124,   365,   371,    -1,   366,   124,   365,   371,
      -1,   366,   402,   124,   365,   371,    -1,   378,   379,    -1,
     378,   213,    15,   371,    21,    -1,   129,    -1,    -1,   371,
      -1,    -1,   371,    -1,    -1,    15,    21,    -1,    15,   375,
      21,    -1,   376,    -1,   375,     8,   376,    -1,   371,    -1,
       5,    -1,    64,    -1,    -1,    -1,    -1,   386,    -1,   387,
      -1,   388,    -1,   418,    -1,   414,    -1,   582,    -1,   423,
      -1,   424,    -1,   425,    -1,   483,    -1,   404,    -1,   419,
      -1,   429,    -1,   216,   493,    -1,   216,   493,   494,   460,
      -1,   123,   492,    -1,   183,   493,    15,   466,    21,    -1,
     394,    -1,   395,    -1,   400,    -1,   397,    -1,   399,    -1,
     415,    -1,   416,    -1,   417,    -1,   382,    -1,   470,    -1,
     468,    -1,   396,    -1,   142,   493,    -1,   142,   493,   371,
      -1,   141,   493,    15,   384,    21,    -1,   140,   493,    15,
      26,    21,    -1,   107,   535,    -1,    10,    -1,   383,    -1,
     385,    -1,    17,    -1,    16,    -1,     5,    -1,     9,    -1,
      37,    -1,    23,    -1,    22,    -1,    35,    -1,    38,    -1,
      34,    -1,    25,    -1,    32,    -1,    29,    -1,    28,    -1,
      31,    -1,    30,    -1,    33,    -1,    24,    -1,   245,   493,
     494,   371,    -1,   245,     8,   493,   378,   393,   494,   371,
      -1,   112,   493,    -1,   112,   493,   371,    -1,   402,   389,
     371,   493,   476,   408,   413,    -1,   388,     8,   371,   476,
     408,   413,    -1,    -1,     7,     7,    -1,     8,   378,   390,
       7,     7,    -1,   391,    -1,   390,     8,   378,   391,    -1,
     183,    -1,   393,    -1,    44,    -1,    87,   476,    -1,   119,
      -1,   145,    15,   392,    21,    -1,   143,    -1,   178,    -1,
     187,    -1,   216,    -1,   230,    -1,   236,    -1,   378,   148,
      -1,   378,   180,    -1,   378,   147,    -1,   194,    -1,   191,
      -1,   145,   493,    15,   392,    21,   494,   371,    -1,   394,
       8,   371,    -1,   178,   493,   494,   371,    -1,   395,     8,
     371,    -1,   230,   493,   494,   422,    -1,   396,     8,   422,
      -1,   191,   493,    -1,   191,   493,   494,   398,   462,    -1,
      -1,   219,   493,    -1,   194,   493,    -1,   194,   493,   494,
     401,   462,    -1,    -1,   406,   403,   410,   403,    -1,   244,
      15,   371,    21,   403,    -1,    -1,   405,   371,    -1,   404,
       8,   371,    -1,    13,    -1,     6,    -1,   407,    -1,   144,
      -1,   200,    -1,    68,    -1,    90,    -1,    91,    -1,   154,
      -1,    63,    -1,    -1,   409,    -1,     5,   556,   513,   557,
     403,    -1,     5,   556,    15,   557,     5,    21,    -1,     5,
     556,    15,   557,   499,    21,    -1,    -1,   409,    -1,    15,
     577,   411,   412,    21,    -1,    15,   577,   411,   412,     8,
     577,   411,   412,    21,    -1,   499,    -1,     5,    -1,   568,
     499,    -1,   568,     5,    -1,    -1,    -1,    26,   499,    -1,
      18,   499,    -1,    87,   494,   493,   371,   476,    -1,   414,
       8,   371,   476,    -1,    44,   493,   494,   422,    -1,   415,
       8,   422,    -1,   187,   493,   494,   422,    -1,   416,     8,
     422,    -1,   236,   493,   494,   422,    -1,   417,     8,   422,
      -1,    67,   493,   422,    -1,    67,   493,   421,   422,    -1,
     418,   550,   421,   550,   422,    -1,   418,     8,   422,    -1,
     167,   493,   420,   502,    -1,   419,   550,   420,   550,   502,
      -1,   419,     8,   502,    -1,    37,   371,    37,    -1,    23,
      -1,    37,   371,    37,    -1,   371,   476,    -1,   119,   493,
     494,   371,    -1,   423,     8,   371,    -1,   143,   493,   494,
     371,    -1,   424,     8,   371,    -1,   117,   493,   426,    -1,
     425,     8,   426,    -1,    15,   427,    21,    -1,   428,     8,
     428,    -1,   427,     8,   428,    -1,   371,    -1,   371,    15,
     498,    21,    -1,   507,    -1,   430,    -1,    80,   492,   431,
     433,    -1,   430,   550,   433,    -1,    -1,    -1,   434,    37,
     435,    37,    -1,   436,    -1,   434,     8,   436,    -1,   447,
      -1,   435,     8,   447,    -1,   437,   439,    -1,   437,   439,
     440,    -1,   437,   439,   441,    -1,   437,   439,   440,   441,
      -1,   437,   444,    -1,    -1,   371,    -1,   371,    -1,    15,
     442,    21,    -1,    15,   443,     7,   443,    21,    -1,   456,
      -1,   442,     8,   456,    -1,    -1,   456,    -1,    15,   445,
       8,   438,    26,   442,    21,    -1,   446,    -1,   445,     8,
     446,    -1,   439,   440,    -1,   439,   441,    -1,   439,   440,
     441,    -1,   444,    -1,   448,    -1,   437,   438,     5,   448,
      -1,   451,     5,   448,    -1,   437,   438,    -1,   450,    -1,
     452,    -1,   454,    -1,    36,    -1,    36,   246,   516,    -1,
      27,    -1,    27,   246,   516,    -1,    64,    -1,   449,    -1,
     437,   502,    15,   577,   495,    21,    -1,    59,    -1,   451,
      -1,    17,   451,    -1,    16,   451,    -1,   149,    -1,   149,
     246,   516,    -1,   453,    -1,    17,   453,    -1,    16,   453,
      -1,   201,    -1,   201,   246,   516,    -1,    92,    -1,    92,
     246,   516,    -1,    15,   455,     8,   455,    21,    -1,   452,
      -1,   450,    -1,   457,    -1,    17,   457,    -1,    16,   457,
      -1,   456,    17,   457,    -1,   456,    16,   457,    -1,   458,
      -1,   457,     5,   458,    -1,   457,    37,   458,    -1,   459,
      -1,   459,     9,   458,    -1,   149,    -1,   438,    -1,    15,
     456,    21,    -1,   461,    -1,   460,     8,   461,    -1,   422,
      -1,   421,    -1,   463,   465,   464,    -1,   462,     8,   463,
     465,   464,    -1,    -1,    -1,   371,    -1,   177,    15,   384,
      21,    -1,    47,    15,    26,    21,    -1,   467,    -1,   466,
       8,   467,    -1,   371,    26,   499,    -1,   163,   469,    -1,
     371,    -1,   469,     8,   371,    -1,   248,   493,   471,    -1,
     248,   493,   471,     8,   380,   474,    -1,   248,   493,   471,
       8,   380,   172,    -1,   248,   493,   471,     8,   380,   172,
     472,    -1,   371,    -1,   473,    -1,   472,     8,   473,    -1,
     475,    -1,   371,    -1,   475,    -1,   474,     8,   475,    -1,
     371,    18,   371,    -1,    -1,    15,   477,    21,    -1,    -1,
     478,   479,    -1,   477,     8,   479,    -1,   480,    -1,     7,
      -1,   499,     7,    -1,   499,     7,   480,    -1,     5,    -1,
     499,    -1,   482,    -1,   481,     8,   482,    -1,   149,    -1,
     130,   493,   484,    -1,   131,    -1,   485,    -1,   484,     8,
     485,    -1,   486,    15,   489,    21,    -1,    -1,   487,   488,
      -1,   231,   407,    -1,   402,    -1,   490,    -1,   489,     8,
     490,    -1,   491,    -1,   491,    16,   491,    -1,   129,    -1,
      -1,    -1,    -1,     7,     7,    -1,    -1,   497,    -1,   499,
      -1,   517,    -1,   568,   499,    -1,   577,   496,    -1,   497,
       8,   577,   496,    -1,   499,    -1,   498,     8,   499,    -1,
     500,    -1,    15,   499,    21,    -1,   515,    -1,   503,    -1,
     511,    -1,   518,    -1,   499,    17,   499,    -1,   499,    16,
     499,    -1,   499,     5,   499,    -1,   499,    37,   499,    -1,
     499,     9,   499,    -1,   383,   499,    -1,    17,   499,    -1,
      16,   499,    -1,   499,    25,   499,    -1,   499,    29,   499,
      -1,   499,    31,   499,    -1,   499,    28,   499,    -1,   499,
      30,   499,    -1,   499,    32,   499,    -1,   499,    24,   499,
      -1,   499,    33,   499,    -1,   499,    38,   499,    -1,   499,
      35,   499,    -1,   499,    22,   499,    -1,    34,   499,    -1,
     499,    23,   499,    -1,   499,   383,   499,    -1,    17,    -1,
      16,    -1,   371,    -1,   502,    -1,   505,    -1,   504,    -1,
     502,    15,   577,   495,    21,    -1,   502,    15,   577,   495,
      21,   509,    -1,   505,    15,   495,    21,    -1,   505,    15,
     495,    21,   509,    -1,   503,     3,   129,    -1,   502,    -1,
     505,    -1,   502,    15,   577,   495,    21,    -1,   505,    15,
     495,    21,    -1,   502,   509,    -1,    -1,   509,    -1,    15,
     510,     7,   510,    21,    -1,    -1,   499,    -1,   512,    -1,
     512,   246,   516,    -1,   513,    -1,   513,   246,   516,    -1,
     514,   508,    -1,    36,    -1,    27,    -1,   201,    -1,    92,
      -1,   149,    -1,    64,    -1,   502,   246,    64,    -1,   513,
     246,    64,    -1,    15,   500,     8,   500,    21,    -1,   502,
      -1,   513,    -1,   499,     7,   499,    -1,   499,     7,    -1,
     499,     7,   499,     7,   499,    -1,   499,     7,     7,   499,
      -1,     7,   499,     7,   499,    -1,     7,     7,   499,    -1,
       7,   499,    -1,     7,    -1,    -1,    -1,    14,   412,   519,
     574,   520,    20,    -1,   502,    -1,   505,    -1,   506,    -1,
     522,     8,   577,   506,    -1,   522,     8,   577,   568,   502,
      -1,   521,    -1,   523,     8,   577,   521,    -1,   523,     8,
     577,   568,   502,    -1,    -1,   502,    -1,   525,     8,   502,
      -1,   547,    -1,   546,    -1,   529,    -1,   537,   529,    -1,
     102,   555,   535,    -1,   103,   555,   534,    -1,   108,   535,
      -1,   527,    -1,   537,   527,    -1,   538,   547,    -1,   538,
     239,    -1,   537,   538,   239,    -1,    97,   555,    15,   499,
      21,   239,   534,    -1,    96,   555,   534,    -1,   106,   555,
     534,    -1,   530,    -1,    76,   555,    -1,   539,   547,    -1,
     539,    -1,   537,   539,    -1,   105,   555,   534,    -1,   583,
      -1,   847,    -1,   865,    -1,    89,   555,    15,   499,    21,
      -1,    89,   555,   556,   545,   557,   617,   528,    -1,    -1,
       8,   378,   254,    15,   499,    21,    -1,   254,    15,   499,
      21,    -1,   185,   555,   556,   545,   557,   550,   543,    26,
     499,     8,   499,    -1,   185,   555,   556,   545,   557,   550,
     543,    26,   499,     8,   499,     8,   499,    -1,    62,   555,
     531,   534,    -1,    84,   555,   534,    -1,   110,   555,   534,
      -1,   218,   555,   378,    62,    15,   499,    21,    -1,   537,
     218,   555,   378,    62,    15,   499,    21,    -1,    15,   533,
      21,    -1,   499,    -1,   499,     7,    -1,     7,   499,    -1,
     499,     7,   499,    -1,   532,    -1,   533,     8,   532,    -1,
      -1,   371,    -1,    -1,   371,    -1,    75,    -1,   536,     7,
      -1,   155,   555,    15,   499,    21,    -1,   122,   555,    15,
     540,   542,    21,    -1,   541,    -1,   540,     8,   541,    -1,
     543,    26,   517,    -1,    -1,     8,   499,    -1,   371,    -1,
     543,    26,   499,     8,   499,    -1,   543,    26,   499,     8,
     499,     8,   499,    -1,    -1,   149,    -1,   113,   555,   534,
      -1,    98,   555,   534,    -1,    98,   555,    15,   499,    21,
     534,    -1,   252,   555,    15,   499,    21,    -1,   537,   252,
     555,    15,   499,    21,    -1,   548,   499,    26,   499,    -1,
     188,   555,   503,    18,   499,    -1,    48,   555,   482,   240,
     371,    -1,    77,   555,    -1,   549,    -1,   558,    -1,    46,
     555,    15,   499,    21,   482,     8,   482,     8,   482,    -1,
     551,    -1,   551,    15,    21,    -1,   551,    15,   552,    21,
      -1,   214,   555,   510,    -1,   554,   555,   510,    -1,    79,
     555,   534,    -1,   115,   555,   534,    -1,    45,   555,    15,
     524,   522,    21,    -1,    81,   555,    15,   524,   523,    21,
      -1,   170,   555,    15,   525,    21,    -1,   253,   555,    15,
     499,    21,   503,    26,   499,    -1,   152,   432,    -1,   186,
     555,   482,    -1,    49,   555,   371,    -1,    49,   555,   371,
     550,    15,   481,    21,    -1,    69,   555,    15,   481,    21,
     550,   499,    -1,    -1,     8,    -1,    61,   555,   371,   577,
      -1,   577,   553,    -1,   552,     8,   577,   553,    -1,   499,
      -1,   568,   499,    -1,     5,   482,    -1,   184,    -1,   232,
      -1,    -1,    -1,    -1,   559,   565,    -1,   559,   580,    -1,
     559,     5,    -1,   559,     9,    -1,   561,   565,    -1,   563,
      -1,   569,   565,    -1,   569,   564,    -1,   569,   565,   572,
      -1,   569,   564,     8,   572,    -1,   570,   565,    -1,   570,
     565,   574,    -1,   571,    -1,   571,     8,   574,    -1,   560,
     555,   578,    -1,    53,    -1,   215,    -1,   104,    -1,   562,
     555,   578,    -1,   176,    -1,    66,    -1,   139,   555,   578,
     565,    -1,   139,   555,   578,   565,   574,    -1,   580,    -1,
       5,    -1,    15,   579,    21,    -1,    15,   566,    21,    -1,
     567,    -1,   566,     8,   577,   567,    -1,   579,    -1,     5,
      -1,     9,    -1,   568,   499,    -1,   568,     5,    -1,   568,
       9,    -1,   166,    -1,   197,   555,   578,    -1,   256,   555,
     578,    -1,   190,   555,   579,   578,    -1,   190,   555,     5,
     578,    -1,   573,    -1,   572,     8,   573,    -1,   503,    -1,
      15,   572,     8,   544,    21,    -1,   500,    -1,   576,    -1,
     575,    -1,   500,     8,   500,    -1,   500,     8,   576,    -1,
     576,     8,   500,    -1,   576,     8,   576,    -1,   575,     8,
     500,    -1,   575,     8,   576,    -1,   515,    -1,    15,   499,
      21,    -1,    15,   500,     8,   544,    21,    -1,    15,   576,
       8,   544,    21,    -1,    15,   575,     8,   544,    21,    -1,
      -1,    -1,   580,    -1,    15,   579,    21,    -1,   503,    -1,
     511,    -1,   579,   501,   579,    -1,   579,     5,   579,    -1,
     579,    37,   579,    -1,   579,     9,   579,    -1,   501,   579,
      -1,   579,    23,   579,    -1,   129,    26,   499,    -1,    -1,
     257,    -1,   584,    -1,   632,    -1,   607,    -1,   586,    -1,
     597,    -1,   592,    -1,   644,    -1,   647,    -1,   725,    -1,
     589,    -1,   598,    -1,   600,    -1,   602,    -1,   604,    -1,
     652,    -1,   658,    -1,   655,    -1,   784,    -1,   782,    -1,
     608,    -1,   633,    -1,   662,    -1,   714,    -1,   712,    -1,
     713,    -1,   715,    -1,   716,    -1,   717,    -1,   718,    -1,
     719,    -1,   727,    -1,   729,    -1,   734,    -1,   731,    -1,
     733,    -1,   737,    -1,   735,    -1,   736,    -1,   748,    -1,
     752,    -1,   753,    -1,   756,    -1,   755,    -1,   757,    -1,
     758,    -1,   759,    -1,   760,    -1,   661,    -1,   742,    -1,
     743,    -1,   744,    -1,   747,    -1,   761,    -1,   764,    -1,
     769,    -1,   774,    -1,   776,    -1,   777,    -1,   778,    -1,
     779,    -1,   781,    -1,   740,    -1,   783,    -1,    82,   493,
     585,    -1,   584,     8,   585,    -1,   371,   476,    -1,    94,
     493,   587,    -1,   588,    -1,   587,     8,   588,    -1,   371,
      -1,   138,   493,   590,    -1,   591,    -1,   590,     8,   591,
      -1,   371,    -1,   228,   493,   596,   593,    -1,    15,   594,
      21,    -1,   595,    -1,   594,     8,   595,    -1,   499,    -1,
     499,     7,   499,    -1,     7,    15,   498,    21,    -1,   371,
      -1,   259,   493,   371,   476,    -1,   303,   493,   371,   476,
      -1,   597,     8,   371,   476,    -1,   136,   493,   599,    -1,
     598,     8,   599,    -1,   371,    -1,   211,   493,   601,    -1,
     600,     8,   601,    -1,   371,    -1,   205,   493,   603,    -1,
     602,     8,   603,    -1,   371,    -1,    70,   493,   605,    -1,
     604,     8,   605,    -1,   371,    -1,   175,   371,   476,    -1,
      -1,    88,   493,   611,   614,   606,    -1,   204,   555,   611,
     615,   617,   606,    -1,   204,   555,   615,   617,   606,     7,
       7,   609,    -1,   610,    -1,   609,     8,   610,    -1,   611,
      -1,   612,    -1,   371,    -1,   371,    15,   498,    21,    -1,
     371,    -1,    -1,   615,   617,    -1,    15,   616,    21,    -1,
     617,   618,    -1,   616,     8,   617,   618,    -1,    -1,    58,
      -1,    58,    15,   577,   631,    21,    -1,   126,    15,   619,
      21,    -1,   258,    15,   619,     8,   499,    21,    -1,   165,
      15,   499,    21,    -1,     5,    -1,   137,    15,   619,    21,
      -1,    86,    15,   620,    21,    -1,   371,    -1,    15,   621,
      21,   378,   255,   623,    -1,   622,    -1,   621,     8,   622,
      -1,   499,    -1,   499,     7,   499,    -1,   624,    -1,   624,
      15,   625,    21,    -1,   371,    -1,   626,    -1,   625,     8,
     626,    -1,   499,    -1,   773,    -1,    40,   627,   628,    -1,
     371,    -1,    -1,   629,    -1,    17,   630,    -1,   628,    17,
     630,    -1,   499,    -1,   568,   499,    -1,   568,   499,     8,
     568,   499,    -1,    43,   493,   635,   637,    -1,   199,   555,
     636,   637,    -1,   199,   555,   637,     7,     7,   634,    -1,
     636,    -1,   634,     8,   636,    -1,   371,    -1,   502,    -1,
      15,   642,    21,   378,   255,   638,    -1,   641,    15,   639,
      21,    -1,   640,    -1,   639,     8,   640,    -1,   499,    -1,
       5,    -1,   517,    -1,   371,    -1,   643,    -1,   642,     8,
     643,    -1,   371,    -1,     5,    -1,     7,    -1,   645,     7,
       7,   493,   371,   476,    -1,   644,     8,   371,   476,    -1,
     646,    -1,   645,     8,   378,   646,    -1,    82,    -1,   259,
      -1,   303,    -1,    94,    -1,    87,    15,   477,    21,    -1,
     228,   593,    -1,    43,    15,   642,    21,   378,   255,   638,
      -1,    43,    -1,    88,   615,   617,   606,    -1,    88,    -1,
      67,    -1,   402,     8,   378,    93,   493,    15,   648,    21,
       7,     7,   650,    -1,    -1,   649,     7,    -1,   648,     8,
       7,    -1,   651,    -1,   650,     8,   651,    -1,   371,    -1,
     127,   493,   653,    -1,   654,    -1,   653,     8,   654,    -1,
     371,    -1,    74,   493,   656,    -1,   657,    -1,   656,     8,
     657,    -1,   371,    -1,    51,   493,   659,    -1,    51,   493,
       8,   378,    67,     7,     7,   659,    -1,   660,    -1,   659,
       8,   660,    -1,   371,   476,    -1,   168,   555,    -1,   182,
     555,    15,   663,    21,   664,   668,    -1,   502,    -1,   663,
       8,   502,    -1,   617,   173,   665,    -1,   617,    -1,   502,
      15,   666,    21,    -1,   667,    -1,   666,     8,   667,    -1,
     499,    -1,     5,    -1,    -1,   669,    -1,   670,    -1,   669,
     670,    -1,   674,    -1,   697,    -1,   705,    -1,   671,    -1,
     681,    -1,   683,    -1,   682,    -1,   672,    -1,   675,    -1,
     676,    -1,   679,    -1,     8,   378,   209,    15,   720,     7,
     721,    21,    -1,     8,   378,   209,    15,   721,    21,    -1,
       8,   378,    71,    15,   673,     7,   721,    21,    -1,     8,
     378,    71,    15,   721,    21,    -1,   371,    -1,     8,   378,
     169,    15,   678,    21,    -1,     8,   378,   282,    15,   678,
      21,    -1,     8,   378,   191,    15,   678,    21,    -1,     8,
     378,   320,    15,   677,    21,    -1,   499,    -1,   499,     8,
     499,    -1,   499,     8,   499,     8,   499,    -1,   503,    -1,
     678,     8,   503,    -1,     8,   378,   322,    15,   680,    21,
      -1,   665,    -1,   680,     8,   665,    -1,     8,   378,   135,
      15,   720,     7,   738,    21,    -1,     8,   378,   135,    15,
     738,    21,    -1,     8,   378,   229,    15,   499,    21,    -1,
       8,   378,    41,   684,    -1,     8,   378,    41,   684,   684,
      -1,    15,   685,   686,   687,    21,    -1,    -1,   148,     7,
      -1,   180,     7,    -1,    -1,   688,    -1,   687,     8,   688,
      -1,   710,    -1,   710,    15,   689,    21,    -1,   710,    15,
     689,    21,    15,   691,    21,    -1,   690,    -1,   689,     8,
     690,    -1,   499,     7,   499,    -1,   692,    -1,   691,     8,
     692,    -1,   693,     7,   694,     7,   695,    -1,   693,     7,
     694,    -1,   693,     7,   695,    -1,   693,    -1,   694,     7,
     695,    -1,   694,    -1,   695,    -1,   378,   217,   696,    -1,
     378,   157,   696,    -1,   378,   128,   696,    -1,    15,   497,
      21,    -1,     8,   378,   208,    15,   698,   702,   699,     8,
     701,    21,    -1,     8,   378,   208,    15,   698,   702,   699,
      21,    -1,     8,   378,   208,    15,   698,   700,   699,     7,
     701,    21,    -1,    -1,    -1,   371,    -1,   378,   702,    -1,
     701,     8,   378,   702,    -1,   703,    15,   503,    21,    -1,
     704,    15,   678,     8,   499,    21,    -1,   234,    -1,   192,
      -1,   162,    -1,   159,    -1,    35,    -1,    22,    -1,    24,
      -1,    33,    -1,   247,    -1,   158,    -1,   161,    -1,     8,
     378,   223,    15,   707,    21,    -1,     8,   378,   224,   706,
      -1,     8,   378,   226,   706,    -1,     8,   378,   221,    -1,
       8,   378,   221,    15,   710,    15,   594,    21,    21,    -1,
     371,    -1,   708,    -1,   707,     8,   708,    -1,   710,    -1,
     710,    15,   709,    78,    21,    -1,   710,    15,   709,   594,
      21,    -1,   710,    15,   709,   594,    21,    15,   378,    78,
      21,    -1,    -1,   502,    -1,   710,    -1,   711,     8,   710,
      -1,   225,   555,   706,    -1,   224,   555,   706,    -1,   227,
     555,   706,    -1,   226,   555,   706,    -1,   222,   555,   706,
      15,   707,    21,    -1,   206,   555,   700,    -1,   207,   555,
     700,    -1,    72,   555,   673,    -1,    73,   555,   673,    -1,
     210,   555,    15,   720,     7,   721,    21,    -1,   210,   555,
      15,   721,    21,    -1,   371,    -1,   722,    -1,   721,     8,
     722,    -1,   710,    15,   723,    21,    -1,   710,    -1,   724,
      -1,   723,     8,   724,    -1,   499,    -1,     7,    -1,   237,
     493,   726,    -1,   725,     8,   726,    -1,   371,   476,    -1,
     238,   555,   728,    -1,   238,   555,   728,   697,    -1,   238,
     555,   728,   672,    -1,   238,   555,   728,   697,   672,    -1,
     238,   555,   728,   672,   697,    -1,   371,    -1,   111,   555,
      -1,   728,    15,   499,    21,    -1,   728,    15,   517,    21,
      -1,   174,   555,   504,   732,    -1,    -1,   675,    -1,   109,
     555,    -1,   160,   555,   730,   378,   175,   613,   476,    -1,
     160,   555,   730,   378,   323,   503,    -1,   189,   555,   720,
      -1,   212,   555,   720,    -1,   135,   555,    15,   720,     7,
     738,    21,    -1,   135,   555,    15,   738,    21,    -1,   739,
      -1,   738,     8,   739,    -1,   710,    -1,   710,    15,   498,
      21,    -1,   134,   555,    -1,   134,   555,   674,    -1,   134,
     555,   741,    -1,   134,   555,   674,   741,    -1,     8,   378,
     208,    15,   678,    21,    -1,    50,   555,   746,    -1,    99,
     555,    -1,    52,   555,   746,    -1,   371,    -1,   745,    -1,
     745,    15,   498,    21,    -1,   120,   555,   503,    26,   503,
      -1,    83,   555,   751,    -1,    83,   555,   751,    15,   749,
      21,    -1,   577,   750,    -1,   749,     8,   577,   750,    -1,
     568,   499,    -1,   149,    -1,   100,   555,   751,    -1,   146,
     555,   754,    -1,    -1,   499,    -1,   332,   555,   498,    -1,
     101,   555,    -1,   241,   555,    -1,   242,   555,    -1,    56,
     555,    -1,    65,   555,   577,    15,   552,    21,   412,   494,
     678,    -1,   324,   555,    15,   762,    21,    -1,   763,    -1,
     762,     8,   763,    -1,   378,   315,    -1,   378,   318,    -1,
     378,   182,    -1,   220,   555,    15,   765,    26,   630,    21,
     617,   768,    -1,   502,    15,   766,    21,    -1,   767,    -1,
     766,     8,   767,    -1,   620,    -1,   773,    -1,   132,   711,
      -1,    -1,   153,   555,    15,   502,    18,   770,    21,    -1,
     502,    -1,   502,    15,   771,    21,    -1,   772,    -1,   771,
       8,   772,    -1,   773,    -1,     7,    -1,     5,    -1,   325,
     555,   499,     8,   378,   330,    15,   678,    21,     8,   378,
     329,    15,   498,    21,   775,    -1,    -1,     8,   378,   182,
      -1,     8,   378,   318,    -1,   326,   555,   499,    -1,   327,
     555,   499,    -1,   327,   555,   499,     8,   378,   315,    -1,
     328,   555,   499,     8,   378,   331,    15,   502,    21,    -1,
     333,   555,    15,   780,    21,    -1,   506,    -1,   780,     8,
     506,    -1,   334,   555,    15,   663,    21,    -1,   833,    -1,
     786,    -1,   785,    -1,   803,    -1,   806,    -1,   807,    -1,
     808,    -1,   809,    -1,   815,    -1,   818,    -1,   823,    -1,
     824,    -1,   825,    -1,   828,    -1,   829,    -1,   830,    -1,
     831,    -1,   832,    -1,   834,    -1,   835,    -1,   836,    -1,
     837,    -1,   838,    -1,   839,    -1,   840,    -1,   841,    -1,
     842,    -1,   268,   555,    -1,   275,   555,    -1,   291,   555,
     617,   787,    -1,   291,   555,   617,    -1,   550,   617,   788,
     617,    -1,   787,   550,   617,   788,   617,    -1,   790,    -1,
     799,    -1,   794,    -1,   795,    -1,   791,    -1,   792,    -1,
     793,    -1,   797,    -1,   798,    -1,   845,    15,   846,   844,
      21,    -1,   191,   789,    -1,   282,   789,    -1,   285,   789,
      -1,   265,   789,    -1,   299,   789,    -1,    84,    15,   378,
     796,    21,    -1,   191,    -1,   299,    -1,   288,    -1,   304,
      15,   499,    21,    -1,   289,    15,   499,    21,    -1,   208,
      15,   800,    21,    -1,   617,   802,     7,   801,    -1,   678,
      -1,    17,    -1,    16,    -1,     5,    -1,    37,    -1,   162,
      -1,   159,    -1,    35,    -1,    22,    -1,    24,    -1,    33,
      -1,   305,    -1,   306,    -1,   307,    -1,   247,    -1,   297,
     555,   617,   804,    -1,   297,   555,   617,    -1,   550,   617,
     805,   617,    -1,   804,   550,   617,   805,   617,    -1,   790,
      -1,   799,    -1,   791,    -1,   792,    -1,   279,   555,   617,
     822,    -1,   279,   555,   617,    -1,   296,   555,    -1,   269,
     555,   617,   810,    -1,   269,   555,   617,    -1,   272,   555,
     617,   822,    -1,   272,   555,   617,    -1,   550,   617,   811,
     617,    -1,   810,   550,   617,   811,   617,    -1,   790,    -1,
     799,    -1,   791,    -1,   792,    -1,   813,    -1,   812,    -1,
     290,    -1,   298,    15,   378,   814,     8,   499,    21,    -1,
     298,    15,   378,   814,    21,    -1,   230,    -1,    94,    -1,
     284,    -1,   295,    -1,   300,   555,   617,   816,    -1,   300,
     555,   617,    -1,   550,   617,   817,   617,    -1,   816,   550,
     617,   817,   617,    -1,   790,    -1,   791,    -1,   280,   555,
     617,   819,    -1,   280,   555,   617,    -1,   550,   617,   820,
     617,    -1,   819,   550,   617,   820,   617,    -1,   822,    -1,
     821,    -1,   266,   789,    -1,   287,    -1,   302,   555,    -1,
     281,   555,   617,   822,    -1,   281,   555,   617,    -1,   292,
     555,   617,   826,    -1,   292,   555,   617,    -1,   550,   617,
     827,   617,    -1,   826,   550,   617,   827,   617,    -1,   790,
      -1,   799,    -1,   794,    -1,   795,    -1,   791,    -1,   792,
      -1,   793,    -1,   797,    -1,   798,    -1,   813,    -1,   812,
      -1,   276,   555,    -1,   293,   555,   617,   787,    -1,   293,
     555,   617,    -1,   277,   555,    -1,   294,   555,   617,   787,
      -1,   294,   555,   617,    -1,   278,   555,    -1,   301,   493,
     789,    -1,   286,   555,    -1,   273,   555,    -1,   290,   555,
      -1,   274,   555,    -1,   264,   555,    -1,   263,   555,    -1,
     283,   555,   789,    -1,   283,   555,    -1,   267,   555,    15,
     502,    21,    -1,   267,   555,    -1,   271,   555,    15,   502,
      21,    -1,   271,   555,    -1,    37,   371,   845,    37,   846,
      -1,   843,    -1,   502,    -1,   844,     8,   843,    -1,   844,
       8,   502,    -1,    -1,    -1,   848,    -1,   861,    -1,   849,
      -1,   862,    -1,   850,    -1,   851,    -1,   863,    -1,   308,
     555,   852,    -1,   310,   555,    -1,   312,   555,    15,   858,
      21,    -1,   312,   555,    15,    21,    -1,   312,   555,    -1,
     313,   555,    15,   858,    21,    -1,   313,   555,    15,    21,
      -1,   313,   555,    -1,   378,   379,    -1,   853,    -1,   854,
      -1,   853,     8,   854,    -1,   378,   855,    -1,   378,   857,
      -1,   378,   856,    -1,   147,    15,   858,    21,    -1,   148,
      15,   858,    21,    -1,   180,    15,   858,    21,    -1,   318,
      15,   858,    21,    -1,   319,    15,   858,    21,    -1,   314,
      15,   859,    21,    -1,   315,    -1,   678,    -1,   860,    -1,
     859,     8,   860,    -1,   378,   316,    -1,   378,   317,    -1,
     309,    -1,   311,    -1,   321,   493,   864,    -1,   378,   379,
      -1,   378,   856,    -1,   866,    -1,   867,    -1,   868,    -1,
     869,    -1,   874,    -1,   894,    -1,   335,   900,    15,   875,
      21,    -1,   336,   900,    15,   884,    21,    -1,   337,   900,
      15,   889,    21,    -1,   339,   900,   892,    -1,   339,   900,
     892,     8,   378,   351,    15,   870,    21,   872,    -1,   339,
     900,   892,     8,   378,   352,    15,   870,    21,   873,    -1,
     871,    -1,   870,     8,   871,    -1,   378,   353,    -1,    -1,
       8,   378,   352,    15,   870,    21,    -1,    -1,     8,   378,
     351,    15,   870,    21,    -1,   340,   900,    -1,   876,    -1,
     875,     8,   876,    -1,   877,    -1,   878,    -1,   879,    -1,
     881,    -1,   880,    -1,   378,   208,    15,   701,    21,    -1,
     378,   191,    15,   678,    21,    -1,   378,   357,    15,   678,
      21,    -1,   378,   356,    15,   513,    21,    -1,   378,   183,
      15,   882,    21,    -1,   883,    -1,   882,     8,   883,    -1,
     506,    26,   499,    -1,   885,    -1,   884,     8,   885,    -1,
     886,    -1,   887,    -1,   888,    -1,   378,   228,    15,   707,
      21,    -1,   378,    41,    15,   707,    21,    -1,   378,   209,
      15,   721,    21,    -1,   890,    -1,   889,     8,   890,    -1,
     378,   338,    -1,   378,   342,    15,   663,    21,    -1,   378,
     341,    -1,   378,   341,    15,   663,    21,    -1,   378,   343,
      15,   893,    21,    -1,   378,   354,    -1,   378,   354,    15,
     891,    21,    -1,   378,   355,    -1,   499,     8,   499,     8,
     499,    -1,   371,    -1,   506,    -1,   893,     8,   506,    -1,
     344,   900,    15,   895,    21,    -1,   896,    -1,   895,     8,
     896,    -1,   378,   244,    15,   897,    21,    -1,   378,   330,
      15,   663,    21,    -1,   378,   345,    15,   663,    21,    -1,
     378,   346,    15,   499,    21,    -1,   378,   347,    15,   899,
       8,   499,    21,    -1,   898,    -1,   897,     8,   898,    -1,
     378,   315,    -1,   378,   350,    -1,   378,   348,    -1,   378,
     349,    -1,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   793,   793,   794,   798,   800,   814,   845,   854,   860,
     880,   889,   905,   917,   927,   934,   940,   945,   950,   974,
    1001,  1015,  1017,  1019,  1023,  1040,  1054,  1078,  1094,  1108,
    1126,  1128,  1135,  1139,  1140,  1147,  1148,  1156,  1157,  1159,
    1163,  1164,  1168,  1172,  1178,  1188,  1192,  1197,  1204,  1205,
    1206,  1207,  1208,  1209,  1210,  1211,  1212,  1213,  1214,  1215,
    1216,  1217,  1222,  1227,  1234,  1236,  1237,  1238,  1239,  1240,
    1241,  1242,  1243,  1244,  1245,  1246,  1247,  1250,  1254,  1262,
    1270,  1279,  1287,  1291,  1293,  1297,  1299,  1301,  1303,  1305,
    1307,  1309,  1311,  1313,  1315,  1317,  1319,  1321,  1323,  1325,
    1327,  1329,  1331,  1336,  1345,  1355,  1363,  1373,  1394,  1414,
    1415,  1417,  1421,  1423,  1427,  1431,  1433,  1437,  1443,  1447,
    1449,  1453,  1457,  1461,  1465,  1469,  1475,  1479,  1483,  1489,
    1494,  1501,  1512,  1525,  1536,  1549,  1559,  1572,  1577,  1584,
    1587,  1592,  1597,  1604,  1607,  1617,  1631,  1634,  1653,  1680,
    1682,  1694,  1702,  1703,  1704,  1705,  1706,  1707,  1708,  1713,
    1714,  1718,  1720,  1727,  1732,  1733,  1735,  1737,  1750,  1756,
    1762,  1771,  1780,  1793,  1794,  1797,  1801,  1816,  1831,  1849,
    1870,  1890,  1912,  1929,  1947,  1954,  1961,  1968,  1981,  1988,
    1995,  2006,  2010,  2012,  2017,  2035,  2046,  2058,  2070,  2084,
    2090,  2097,  2103,  2109,  2117,  2124,  2140,  2143,  2152,  2154,
    2158,  2162,  2182,  2186,  2188,  2192,  2193,  2196,  2198,  2200,
    2202,  2204,  2207,  2210,  2214,  2220,  2224,  2228,  2230,  2235,
    2236,  2240,  2244,  2246,  2250,  2252,  2254,  2259,  2263,  2265,
    2267,  2270,  2272,  2273,  2274,  2275,  2276,  2277,  2278,  2279,
    2282,  2283,  2289,  2292,  2293,  2295,  2299,  2300,  2303,  2304,
    2306,  2310,  2311,  2312,  2313,  2315,  2318,  2319,  2328,  2330,
    2337,  2344,  2351,  2360,  2362,  2364,  2368,  2370,  2374,  2383,
    2390,  2397,  2399,  2403,  2407,  2413,  2415,  2420,  2424,  2428,
    2435,  2442,  2452,  2454,  2458,  2470,  2473,  2482,  2495,  2501,
    2507,  2513,  2521,  2531,  2533,  2537,  2539,  2572,  2574,  2578,
    2617,  2618,  2622,  2622,  2627,  2631,  2639,  2648,  2657,  2667,
    2673,  2676,  2678,  2682,  2690,  2705,  2712,  2714,  2718,  2734,
    2734,  2738,  2740,  2752,  2754,  2758,  2764,  2776,  2788,  2805,
    2834,  2835,  2843,  2844,  2848,  2850,  2852,  2863,  2867,  2873,
    2875,  2879,  2881,  2883,  2887,  2889,  2893,  2895,  2897,  2899,
    2901,  2903,  2905,  2907,  2909,  2911,  2913,  2915,  2917,  2919,
    2921,  2923,  2925,  2927,  2929,  2931,  2933,  2935,  2937,  2941,
    2942,  2953,  3027,  3039,  3041,  3045,  3176,  3226,  3270,  3312,
    3370,  3372,  3374,  3413,  3456,  3467,  3468,  3472,  3477,  3478,
    3482,  3484,  3490,  3492,  3498,  3511,  3517,  3524,  3530,  3538,
    3546,  3562,  3572,  3585,  3592,  3594,  3617,  3619,  3621,  3623,
    3625,  3627,  3629,  3631,  3635,  3635,  3635,  3649,  3651,  3674,
    3676,  3678,  3694,  3696,  3698,  3712,  3715,  3717,  3725,  3727,
    3729,  3731,  3785,  3805,  3820,  3829,  3832,  3882,  3888,  3893,
    3911,  3913,  3915,  3917,  3919,  3922,  3928,  3930,  3932,  3935,
    3937,  3939,  3966,  3975,  3984,  3985,  3987,  3992,  3999,  4007,
    4009,  4013,  4016,  4018,  4022,  4028,  4030,  4032,  4034,  4038,
    4040,  4049,  4050,  4057,  4058,  4062,  4066,  4087,  4090,  4094,
    4096,  4103,  4108,  4109,  4120,  4132,  4155,  4180,  4181,  4188,
    4190,  4192,  4194,  4196,  4200,  4277,  4289,  4296,  4298,  4299,
    4301,  4310,  4317,  4324,  4332,  4337,  4342,  4345,  4348,  4351,
    4354,  4357,  4361,  4379,  4384,  4403,  4422,  4426,  4427,  4430,
    4434,  4439,  4446,  4448,  4450,  4454,  4455,  4466,  4481,  4485,
    4492,  4495,  4505,  4518,  4531,  4534,  4536,  4539,  4542,  4546,
    4555,  4558,  4562,  4564,  4570,  4574,  4576,  4578,  4585,  4589,
    4591,  4595,  4597,  4601,  4620,  4636,  4645,  4654,  4656,  4660,
    4686,  4701,  4716,  4733,  4741,  4750,  4758,  4763,  4768,  4790,
    4806,  4808,  4812,  4814,  4821,  4823,  4825,  4829,  4831,  4833,
    4835,  4837,  4839,  4843,  4846,  4849,  4855,  4861,  4870,  4874,
    4881,  4883,  4887,  4889,  4891,  4896,  4901,  4906,  4911,  4920,
    4925,  4931,  4932,  4947,  4948,  4949,  4950,  4951,  4952,  4953,
    4954,  4955,  4956,  4957,  4958,  4959,  4960,  4961,  4962,  4963,
    4964,  4965,  4968,  4969,  4970,  4971,  4972,  4973,  4974,  4975,
    4976,  4977,  4978,  4979,  4980,  4981,  4982,  4983,  4984,  4985,
    4986,  4987,  4988,  4989,  4990,  4991,  4992,  4993,  4994,  4995,
    4996,  4997,  4998,  4999,  5000,  5001,  5002,  5003,  5004,  5005,
    5006,  5007,  5008,  5009,  5010,  5011,  5015,  5017,  5028,  5049,
    5053,  5055,  5059,  5072,  5076,  5078,  5082,  5093,  5104,  5108,
    5110,  5114,  5116,  5118,  5133,  5145,  5165,  5185,  5207,  5213,
    5222,  5230,  5236,  5244,  5251,  5257,  5266,  5270,  5276,  5284,
    5298,  5312,  5317,  5333,  5348,  5376,  5378,  5382,  5384,  5388,
    5417,  5440,  5461,  5462,  5466,  5487,  5489,  5493,  5501,  5505,
    5510,  5512,  5514,  5516,  5522,  5524,  5528,  5538,  5542,  5544,
    5549,  5551,  5555,  5559,  5565,  5575,  5577,  5581,  5583,  5585,
    5592,  5610,  5611,  5615,  5617,  5621,  5628,  5638,  5667,  5682,
    5689,  5707,  5709,  5713,  5727,  5753,  5766,  5782,  5784,  5787,
    5789,  5795,  5799,  5827,  5829,  5833,  5841,  5847,  5850,  5908,
    5972,  5974,  5977,  5981,  5985,  5989,  6006,  6018,  6022,  6026,
    6036,  6041,  6046,  6053,  6062,  6062,  6073,  6084,  6086,  6090,
    6101,  6105,  6107,  6111,  6122,  6126,  6128,  6132,  6144,  6146,
    6153,  6155,  6159,  6175,  6183,  6194,  6196,  6200,  6203,  6208,
    6218,  6220,  6224,  6226,  6235,  6236,  6240,  6242,  6247,  6248,
    6249,  6250,  6251,  6252,  6253,  6254,  6255,  6256,  6257,  6260,
    6265,  6269,  6273,  6277,  6290,  6294,  6298,  6302,  6305,  6307,
    6309,  6313,  6315,  6319,  6323,  6325,  6329,  6334,  6338,  6342,
    6344,  6348,  6357,  6360,  6366,  6373,  6376,  6378,  6382,  6384,
    6388,  6400,  6402,  6406,  6410,  6412,  6416,  6418,  6420,  6422,
    6424,  6426,  6428,  6432,  6436,  6440,  6444,  6448,  6455,  6461,
    6466,  6469,  6472,  6485,  6487,  6491,  6493,  6498,  6504,  6510,
    6516,  6522,  6528,  6534,  6540,  6546,  6555,  6561,  6578,  6580,
    6588,  6596,  6598,  6602,  6606,  6608,  6612,  6614,  6622,  6626,
    6638,  6641,  6659,  6661,  6665,  6667,  6671,  6673,  6677,  6681,
    6685,  6694,  6698,  6702,  6707,  6711,  6723,  6725,  6729,  6734,
    6738,  6740,  6744,  6746,  6750,  6755,  6762,  6785,  6787,  6789,
    6791,  6793,  6797,  6808,  6812,  6827,  6834,  6841,  6842,  6846,
    6850,  6858,  6862,  6866,  6874,  6879,  6893,  6895,  6899,  6901,
    6910,  6912,  6914,  6916,  6952,  6956,  6960,  6964,  6968,  6980,
    6982,  6986,  6989,  6991,  6995,  7000,  7007,  7010,  7018,  7022,
    7027,  7029,  7036,  7041,  7045,  7049,  7053,  7057,  7061,  7064,
    7066,  7070,  7072,  7074,  7078,  7082,  7094,  7096,  7100,  7102,
    7106,  7109,  7112,  7116,  7122,  7134,  7136,  7140,  7142,  7146,
    7154,  7166,  7167,  7169,  7173,  7177,  7179,  7187,  7191,  7194,
    7196,  7200,  7204,  7206,  7207,  7208,  7209,  7210,  7211,  7212,
    7213,  7214,  7215,  7216,  7217,  7218,  7219,  7220,  7221,  7222,
    7223,  7224,  7225,  7226,  7227,  7228,  7229,  7230,  7231,  7234,
    7240,  7246,  7252,  7258,  7262,  7268,  7269,  7270,  7271,  7272,
    7273,  7274,  7275,  7276,  7279,  7284,  7289,  7295,  7301,  7307,
    7312,  7318,  7324,  7330,  7337,  7343,  7349,  7356,  7360,  7362,
    7368,  7375,  7381,  7387,  7393,  7399,  7405,  7411,  7417,  7423,
    7429,  7435,  7441,  7451,  7456,  7462,  7466,  7472,  7473,  7474,
    7475,  7478,  7486,  7492,  7498,  7503,  7509,  7516,  7522,  7526,
    7532,  7533,  7534,  7535,  7536,  7537,  7540,  7549,  7553,  7559,
    7566,  7573,  7580,  7589,  7595,  7601,  7605,  7611,  7612,  7615,
    7621,  7627,  7631,  7638,  7639,  7642,  7648,  7654,  7659,  7667,
    7673,  7678,  7685,  7689,  7695,  7696,  7697,  7698,  7699,  7700,
    7701,  7702,  7703,  7704,  7705,  7709,  7714,  7719,  7726,  7731,
    7737,  7743,  7748,  7753,  7758,  7762,  7767,  7772,  7776,  7781,
    7785,  7791,  7796,  7802,  7807,  7813,  7823,  7827,  7831,  7835,
    7841,  7844,  7848,  7849,  7850,  7851,  7852,  7853,  7854,  7857,
    7861,  7865,  7867,  7869,  7873,  7875,  7877,  7881,  7883,  7887,
    7889,  7893,  7896,  7899,  7904,  7906,  7908,  7910,  7912,  7916,
    7920,  7925,  7929,  7931,  7935,  7937,  7941,  7945,  7949,  7953,
    7955,  7959,  7960,  7961,  7962,  7963,  7964,  7967,  7971,  7975,
    7979,  7981,  7983,  7987,  7989,  7993,  7998,  7999,  8004,  8005,
    8009,  8013,  8015,  8019,  8020,  8021,  8022,  8023,  8026,  8030,
    8034,  8038,  8042,  8045,  8047,  8051,  8055,  8057,  8061,  8062,
    8063,  8066,  8070,  8074,  8078,  8080,  8084,  8086,  8088,  8090,
    8093,  8095,  8097,  8099,  8103,  8110,  8114,  8116,  8120,  8124,
    8126,  8130,  8132,  8134,  8136,  8138,  8142,  8144,  8148,  8150,
    8154,  8156,  8161
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "PERCENT", "AMPERSAND", "ASTER",
  "CLUSTER", "COLON", "COMMA", "DASTER", "DEFINED_OPERATOR", "DOT",
  "DQUOTE", "GLOBAL_A", "LEFTAB", "LEFTPAR", "MINUS", "PLUS", "POINT_TO",
  "QUOTE", "RIGHTAB", "RIGHTPAR", "AND", "DSLASH", "EQV", "EQ", "EQUAL",
  "FFALSE", "GE", "GT", "LE", "LT", "NE", "NEQV", "NOT", "OR", "TTRUE",
  "SLASH", "XOR", "REFERENCE", "AT", "ACROSS", "ALIGN_WITH", "ALIGN",
  "ALLOCATABLE", "ALLOCATE", "ARITHIF", "ASSIGNMENT", "ASSIGN",
  "ASSIGNGOTO", "ASYNCHRONOUS", "ASYNCID", "ASYNCWAIT", "BACKSPACE",
  "BAD_CCONST", "BAD_SYMBOL", "BARRIER", "BLOCKDATA", "BLOCK",
  "BOZ_CONSTANT", "BYTE", "CALL", "CASE", "CHARACTER", "CHAR_CONSTANT",
  "CHECK", "CLOSE", "COMMON", "COMPLEX", "COMPGOTO", "CONSISTENT_GROUP",
  "CONSISTENT_SPEC", "CONSISTENT_START", "CONSISTENT_WAIT", "CONSISTENT",
  "CONSTRUCT_ID", "CONTAINS", "CONTINUE", "CORNER", "CYCLE", "DATA",
  "DEALLOCATE", "HPF_TEMPLATE", "DEBUG", "DEFAULT_CASE", "DEFINE",
  "DERIVED", "DIMENSION", "DISTRIBUTE", "DOWHILE", "DOUBLEPRECISION",
  "DOUBLECOMPLEX", "DP_CONSTANT", "DVM_POINTER", "DYNAMIC", "ELEMENTAL",
  "ELSE", "ELSEIF", "ELSEWHERE", "ENDASYNCHRONOUS", "ENDDEBUG",
  "ENDINTERVAL", "ENDUNIT", "ENDDO", "ENDFILE", "ENDFORALL", "ENDIF",
  "ENDINTERFACE", "ENDMODULE", "ENDON", "ENDSELECT", "ENDTASK_REGION",
  "ENDTYPE", "ENDWHERE", "ENTRY", "EXIT", "EOLN", "EQUIVALENCE", "ERROR",
  "EXTERNAL", "F90", "FIND", "FORALL", "FORMAT", "FUNCTION", "GATE",
  "GEN_BLOCK", "HEAP", "HIGH", "IDENTIFIER", "IMPLICIT", "IMPLICITNONE",
  "INCLUDE_TO", "INCLUDE", "INDEPENDENT", "INDIRECT_ACCESS",
  "INDIRECT_GROUP", "INDIRECT", "INHERIT", "INQUIRE",
  "INTERFACEASSIGNMENT", "INTERFACEOPERATOR", "INTERFACE", "INTRINSIC",
  "INTEGER", "INTENT", "INTERVAL", "INOUT", "IN", "INT_CONSTANT", "LABEL",
  "LABEL_DECLARE", "LET", "LOCALIZE", "LOGICAL", "LOGICALIF", "LOOP",
  "LOW", "MAXLOC", "MAX", "MAP", "MINLOC", "MIN", "MODULE_PROCEDURE",
  "MODULE", "MULT_BLOCK", "NAMEEQ", "NAMELIST", "NEW_VALUE", "NEW",
  "NULLIFY", "OCTAL_CONSTANT", "ONLY", "ON", "ON_DIR", "ONTO", "OPEN",
  "OPERATOR", "OPTIONAL", "OTHERWISE", "OUT", "OWN", "PARALLEL",
  "PARAMETER", "PAUSE", "PLAINDO", "PLAINGOTO", "POINTER", "POINTERLET",
  "PREFETCH", "PRINT", "PRIVATE", "PRODUCT", "PROGRAM", "PUBLIC", "PURE",
  "RANGE", "READ", "REALIGN_WITH", "REALIGN", "REAL", "REAL_CONSTANT",
  "RECURSIVE", "REDISTRIBUTE_NEW", "REDISTRIBUTE", "REDUCTION_GROUP",
  "REDUCTION_START", "REDUCTION_WAIT", "REDUCTION", "REMOTE_ACCESS_SPEC",
  "REMOTE_ACCESS", "REMOTE_GROUP", "RESET", "RESULT", "RETURN", "REWIND",
  "SAVE", "SECTION", "SELECT", "SEQUENCE", "SHADOW_ADD", "SHADOW_COMPUTE",
  "SHADOW_GROUP", "SHADOW_RENEW", "SHADOW_START_SPEC", "SHADOW_START",
  "SHADOW_WAIT_SPEC", "SHADOW_WAIT", "SHADOW", "STAGE", "STATIC", "STAT",
  "STOP", "SUBROUTINE", "SUM", "SYNC", "TARGET", "TASK", "TASK_REGION",
  "THEN", "TO", "TRACEON", "TRACEOFF", "TRUNC", "TYPE", "TYPE_DECL",
  "UNDER", "UNKNOWN", "USE", "VIRTUAL", "VARIABLE", "WAIT", "WHERE",
  "WHERE_ASSIGN", "WHILE", "WITH", "WRITE", "COMMENT", "WGT_BLOCK",
  "HPF_PROCESSORS", "IOSTAT", "ERR", "END", "OMPDVM_ATOMIC",
  "OMPDVM_BARRIER", "OMPDVM_COPYIN", "OMPDVM_COPYPRIVATE",
  "OMPDVM_CRITICAL", "OMPDVM_ONETHREAD", "OMPDVM_DO", "OMPDVM_DYNAMIC",
  "OMPDVM_ENDCRITICAL", "OMPDVM_ENDDO", "OMPDVM_ENDMASTER",
  "OMPDVM_ENDORDERED", "OMPDVM_ENDPARALLEL", "OMPDVM_ENDPARALLELDO",
  "OMPDVM_ENDPARALLELSECTIONS", "OMPDVM_ENDPARALLELWORKSHARE",
  "OMPDVM_ENDSECTIONS", "OMPDVM_ENDSINGLE", "OMPDVM_ENDWORKSHARE",
  "OMPDVM_FIRSTPRIVATE", "OMPDVM_FLUSH", "OMPDVM_GUIDED",
  "OMPDVM_LASTPRIVATE", "OMPDVM_MASTER", "OMPDVM_NOWAIT", "OMPDVM_NONE",
  "OMPDVM_NUM_THREADS", "OMPDVM_ORDERED", "OMPDVM_PARALLEL",
  "OMPDVM_PARALLELDO", "OMPDVM_PARALLELSECTIONS",
  "OMPDVM_PARALLELWORKSHARE", "OMPDVM_RUNTIME", "OMPDVM_SECTION",
  "OMPDVM_SECTIONS", "OMPDVM_SCHEDULE", "OMPDVM_SHARED", "OMPDVM_SINGLE",
  "OMPDVM_THREADPRIVATE", "OMPDVM_WORKSHARE", "OMPDVM_NODES", "OMPDVM_IF",
  "IAND", "IEOR", "IOR", "ACC_REGION", "ACC_END_REGION",
  "ACC_CHECKSECTION", "ACC_END_CHECKSECTION", "ACC_GET_ACTUAL",
  "ACC_ACTUAL", "ACC_TARGETS", "ACC_ASYNC", "ACC_HOST", "ACC_CUDA",
  "ACC_LOCAL", "ACC_INLOCAL", "ACC_CUDA_BLOCK", "ACC_ROUTINE", "ACC_TIE",
  "BY", "IO_MODE", "CP_CREATE", "CP_LOAD", "CP_SAVE", "CP_WAIT", "FILES",
  "VARLIST", "STATUS", "EXITINTERVAL", "TEMPLATE_CREATE",
  "TEMPLATE_DELETE", "SPF_ANALYSIS", "SPF_PARALLEL", "SPF_TRANSFORM",
  "SPF_NOINLINE", "SPF_PARALLEL_REG", "SPF_END_PARALLEL_REG", "SPF_EXPAND",
  "SPF_FISSION", "SPF_SHRINK", "SPF_CHECKPOINT", "SPF_EXCEPT",
  "SPF_FILES_COUNT", "SPF_INTERVAL", "SPF_TIME", "SPF_ITER",
  "SPF_FLEXIBLE", "SPF_APPLY_REGION", "SPF_APPLY_FRAGMENT",
  "SPF_CODE_COVERAGE", "SPF_UNROLL", "SPF_MERGE", "SPF_COVER",
  "SPF_PROCESS_PRIVATE", "BINARY_OP", "UNARY_OP", "$accept", "program",
  "stat", "thislabel", "entry", "new_prog", "proc_attr", "procname",
  "funcname", "typedfunc", "opt_result_clause", "name", "progname",
  "blokname", "arglist", "args", "arg", "filename", "needkeyword",
  "keywordoff", "keyword_if_colon_follow", "spec", "interface",
  "defined_op", "operator", "intrinsic_op", "type_dcl", "end_type", "dcl",
  "options", "attr_spec_list", "attr_spec", "intent_spec", "access_spec",
  "intent", "optional", "static", "private", "private_attr", "sequence",
  "public", "public_attr", "type", "opt_key_hedr", "attrib", "att_type",
  "typespec", "typename", "lengspec", "proper_lengspec", "selector",
  "clause", "end_ioctl", "initial_value", "dimension", "allocatable",
  "pointer", "target", "common", "namelist", "namelist_group", "comblock",
  "var", "external", "intrinsic", "equivalence", "equivset", "equivlist",
  "equi_object", "data", "data1", "data_in", "in_data", "datapair",
  "datalvals", "datarvals", "datalval", "data_null", "d_name", "dataname",
  "datasubs", "datarange", "iconexprlist", "opticonexpr", "dataimplieddo",
  "dlist", "dataelt", "datarval", "datavalue", "BOZ_const", "int_const",
  "unsignedint", "real_const", "unsignedreal", "complex_const_data",
  "complex_part", "iconexpr", "iconterm", "iconfactor", "iconprimary",
  "savelist", "saveitem", "use_name_list", "use_key_word",
  "no_use_key_word", "use_name", "paramlist", "paramitem",
  "module_proc_stmt", "proc_name_list", "use_stat", "module_name",
  "only_list", "only_name", "rename_list", "rename_name", "dims",
  "dimlist", "@1", "dim", "ubound", "labellist", "label", "implicit",
  "implist", "impitem", "imptype", "@2", "type_implicit", "letgroups",
  "letgroup", "letter", "inside", "in_dcl", "opt_double_colon",
  "funarglist", "funarg", "funargs", "subscript_list", "expr", "uexpr",
  "addop", "ident", "lhs", "array_ele_substring_func_ref",
  "structure_component", "array_element", "asubstring", "opt_substring",
  "substring", "opt_expr", "simple_const", "numeric_bool_const",
  "integer_constant", "string_constant", "complex_const", "kind",
  "triplet", "vec", "@3", "@4", "allocate_object", "allocation_list",
  "allocate_object_list", "stat_spec", "pointer_name_list", "exec",
  "do_while", "opt_while", "plain_do", "case", "case_selector",
  "case_value_range", "case_value_range_list", "opt_construct_name",
  "opt_unit_name", "construct_name", "construct_name_colon", "logif",
  "forall", "forall_list", "forall_expr", "opt_forall_cond", "do_var",
  "dospec", "dotarget", "whereable", "iffable", "let", "goto", "opt_comma",
  "call", "callarglist", "callarg", "stop", "end_spec", "intonlyon",
  "intonlyoff", "io", "iofmove", "fmkwd", "iofctl", "ctlkwd", "inquire",
  "infmt", "ioctl", "ctllist", "ioclause", "nameeq", "read", "write",
  "print", "inlist", "inelt", "outlist", "out2", "other", "in_ioctl",
  "start_ioctl", "fexpr", "unpar_fexpr", "cmnt", "dvm_specification",
  "dvm_exec", "dvm_template", "template_obj", "dvm_dynamic",
  "dyn_array_name_list", "dyn_array_name", "dvm_inherit",
  "dummy_array_name_list", "dummy_array_name", "dvm_shadow",
  "shadow_attr_stuff", "sh_width_list", "sh_width", "sh_array_name",
  "dvm_processors", "dvm_indirect_group", "indirect_group_name",
  "dvm_remote_group", "remote_group_name", "dvm_reduction_group",
  "reduction_group_name", "dvm_consistent_group", "consistent_group_name",
  "opt_onto", "dvm_distribute", "dvm_redistribute", "dist_name_list",
  "distributee", "dist_name", "pointer_ar_elem", "processors_name",
  "opt_dist_format_clause", "dist_format_clause", "dist_format_list",
  "opt_key_word", "dist_format", "array_name", "derived_spec",
  "derived_elem_list", "derived_elem", "target_spec", "derived_target",
  "derived_subscript_list", "derived_subscript", "dummy_ident",
  "opt_plus_shadow", "plus_shadow", "shadow_id", "shadow_width",
  "dvm_align", "dvm_realign", "realignee_list", "alignee", "realignee",
  "align_directive_stuff", "align_base", "align_subscript_list",
  "align_subscript", "align_base_name", "dim_ident_list", "dim_ident",
  "dvm_combined_dir", "dvm_attribute_list", "dvm_attribute", "dvm_pointer",
  "dimension_list", "@5", "pointer_var_list", "pointer_var", "dvm_heap",
  "heap_array_name_list", "heap_array_name", "dvm_consistent",
  "consistent_array_name_list", "consistent_array_name", "dvm_asyncid",
  "async_id_list", "async_id", "dvm_new_value", "dvm_parallel_on",
  "ident_list", "opt_on", "distribute_cycles", "par_subscript_list",
  "par_subscript", "opt_spec", "spec_list", "par_spec",
  "remote_access_spec", "consistent_spec", "consistent_group", "new_spec",
  "private_spec", "cuda_block_spec", "sizelist", "variable_list",
  "tie_spec", "tied_array_list", "indirect_access_spec", "stage_spec",
  "across_spec", "in_out_across", "opt_keyword_in_out", "opt_in_out",
  "dependent_array_list", "dependent_array", "dependence_list",
  "dependence", "section_spec_list", "section_spec", "ar_section",
  "low_section", "high_section", "section", "reduction_spec",
  "opt_key_word_r", "no_opt_key_word_r", "reduction_group",
  "reduction_list", "reduction", "reduction_op", "loc_op", "shadow_spec",
  "shadow_group_name", "shadow_list", "shadow", "opt_corner",
  "array_ident", "array_ident_list", "dvm_shadow_start", "dvm_shadow_wait",
  "dvm_shadow_group", "dvm_reduction_start", "dvm_reduction_wait",
  "dvm_consistent_start", "dvm_consistent_wait", "dvm_remote_access",
  "group_name", "remote_data_list", "remote_data", "remote_index_list",
  "remote_index", "dvm_task", "task_array", "dvm_task_region", "task_name",
  "dvm_end_task_region", "task", "dvm_on", "opt_private_spec",
  "dvm_end_on", "dvm_map", "dvm_prefetch", "dvm_reset",
  "dvm_indirect_access", "indirect_list", "indirect_reference",
  "hpf_independent", "hpf_reduction_spec", "dvm_asynchronous",
  "dvm_endasynchronous", "dvm_asyncwait", "async_ident", "async",
  "dvm_f90", "dvm_debug_dir", "debparamlist", "debparam",
  "fragment_number", "dvm_enddebug_dir", "dvm_interval_dir",
  "interval_number", "dvm_exit_interval_dir", "dvm_endinterval_dir",
  "dvm_traceon_dir", "dvm_traceoff_dir", "dvm_barrier_dir", "dvm_check",
  "dvm_io_mode_dir", "mode_list", "mode_spec", "dvm_shadow_add",
  "template_ref", "shadow_axis_list", "shadow_axis", "opt_include_to",
  "dvm_localize", "localize_target", "target_subscript_list",
  "target_subscript", "aster_expr", "dvm_cp_create", "opt_mode",
  "dvm_cp_load", "dvm_cp_save", "dvm_cp_wait", "dvm_template_create",
  "template_list", "dvm_template_delete", "omp_specification_directive",
  "omp_execution_directive", "ompdvm_onethread",
  "omp_parallel_end_directive", "omp_parallel_begin_directive",
  "parallel_clause_list", "parallel_clause", "omp_variable_list_in_par",
  "ompprivate_clause", "ompfirstprivate_clause", "omplastprivate_clause",
  "ompcopyin_clause", "ompshared_clause", "ompdefault_clause", "def_expr",
  "ompif_clause", "ompnumthreads_clause", "ompreduction_clause",
  "ompreduction", "ompreduction_vars", "ompreduction_op",
  "omp_sections_begin_directive", "sections_clause_list",
  "sections_clause", "omp_sections_end_directive", "omp_section_directive",
  "omp_do_begin_directive", "omp_do_end_directive", "do_clause_list",
  "do_clause", "ompordered_clause", "ompschedule_clause", "ompschedule_op",
  "omp_single_begin_directive", "single_clause_list", "single_clause",
  "omp_single_end_directive", "end_single_clause_list",
  "end_single_clause", "ompcopyprivate_clause", "ompnowait_clause",
  "omp_workshare_begin_directive", "omp_workshare_end_directive",
  "omp_parallel_do_begin_directive", "paralleldo_clause_list",
  "paralleldo_clause", "omp_parallel_do_end_directive",
  "omp_parallel_sections_begin_directive",
  "omp_parallel_sections_end_directive",
  "omp_parallel_workshare_begin_directive",
  "omp_parallel_workshare_end_directive", "omp_threadprivate_directive",
  "omp_master_begin_directive", "omp_master_end_directive",
  "omp_ordered_begin_directive", "omp_ordered_end_directive",
  "omp_barrier_directive", "omp_atomic_directive", "omp_flush_directive",
  "omp_critical_begin_directive", "omp_critical_end_directive",
  "omp_common_var", "omp_variable_list", "op_slash_1", "op_slash_0",
  "acc_directive", "acc_region", "acc_checksection", "acc_get_actual",
  "acc_actual", "opt_clause", "acc_clause_list", "acc_clause",
  "data_clause", "targets_clause", "async_clause", "acc_var_list",
  "computer_list", "computer", "acc_end_region", "acc_end_checksection",
  "acc_routine", "opt_targets_clause", "spf_directive", "spf_analysis",
  "spf_parallel", "spf_transform", "spf_parallel_reg",
  "characteristic_list", "characteristic", "opt_clause_apply_fragment",
  "opt_clause_apply_region", "spf_end_parallel_reg", "analysis_spec_list",
  "analysis_spec", "analysis_reduction_spec", "analysis_private_spec",
  "analysis_process_private_spec", "analysis_cover_spec",
  "analysis_parameter_spec", "spf_parameter_list", "spf_parameter",
  "parallel_spec_list", "parallel_spec", "parallel_shadow_spec",
  "parallel_across_spec", "parallel_remote_access_spec",
  "transform_spec_list", "transform_spec", "unroll_list", "region_name",
  "array_element_list", "spf_checkpoint", "checkpoint_spec_list",
  "checkpoint_spec", "spf_type_list", "spf_type", "interval_spec",
  "in_unit", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   356,   357,     1,     2,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,   111,   112,   113,   114,   115,   116,   117,
     118,   119,   120,   121,   122,   123,   124,   125,   126,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
     138,   139,   140,   141,   142,   143,   144,   145,   146,   147,
     148,   149,   150,   151,   152,   153,   154,   155,   156,   157,
     158,   159,   160,   161,   162,   163,   164,   165,   166,   167,
     168,   169,   170,   171,   172,   173,   174,   175,   176,   177,
     178,   179,   180,   181,   182,   183,   184,   185,   186,   187,
     188,   189,   190,   191,   192,   193,   194,   195,   196,   197,
     198,   199,   200,   201,   202,   203,   204,   205,   206,   207,
     208,   209,   210,   211,   212,   213,   214,   215,   216,   217,
     218,   219,   220,   221,   222,   223,   224,   225,   226,   227,
     228,   229,   230,   231,   232,   233,   234,   235,   236,   237,
     238,   239,   240,   241,   242,   243,   244,   245,   246,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   286,   287,
     288,   289,   290,   291,   292,   293,   294,   295,   296,   297,
     298,   299,   300,   301,   302,   303,   304,   305,   306,   307,
     308,   309,   310,   311,   312,   313,   314,   315,   316,   317,
     318,   319,   320,   321,   322,   323,   324,   325,   326,   327,
     328,   329,   330,   331,   332,   333,   334,   335,   336,   337,
     338,   339,   340,   341,   342,   343,   344,   345,   346,   347,
     348,   349,   350,   351,   352,   353,   354,   355,   358,   359
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   360,   361,   361,   362,   362,   362,   362,   362,   362,
     362,   363,   364,   364,   364,   364,   364,   364,   364,   364,
     365,   366,   366,   366,   367,   368,   369,   369,   369,   369,
     370,   370,   371,   372,   372,   373,   373,   374,   374,   374,
     375,   375,   376,   376,   377,   378,   379,   380,   381,   381,
     381,   381,   381,   381,   381,   381,   381,   381,   381,   381,
     381,   381,   381,   381,   381,   381,   381,   381,   381,   381,
     381,   381,   381,   381,   381,   381,   381,   382,   382,   382,
     382,   382,   383,   384,   384,   385,   385,   385,   385,   385,
     385,   385,   385,   385,   385,   385,   385,   385,   385,   385,
     385,   385,   385,   386,   386,   387,   387,   388,   388,   389,
     389,   389,   390,   390,   391,   391,   391,   391,   391,   391,
     391,   391,   391,   391,   391,   391,   392,   392,   392,   393,
     393,   394,   394,   395,   395,   396,   396,   397,   397,   398,
     399,   400,   400,   401,   402,   402,   403,   404,   404,   405,
     405,   406,   407,   407,   407,   407,   407,   407,   407,   408,
     408,   409,   409,   409,   410,   410,   410,   410,   411,   411,
     411,   411,   412,   413,   413,   413,   414,   414,   415,   415,
     416,   416,   417,   417,   418,   418,   418,   418,   419,   419,
     419,   420,   421,   421,   422,   423,   423,   424,   424,   425,
     425,   426,   427,   427,   428,   428,   428,   429,   430,   430,
     431,   432,   433,   434,   434,   435,   435,   436,   436,   436,
     436,   436,   437,   438,   439,   440,   441,   442,   442,   443,
     443,   444,   445,   445,   446,   446,   446,   446,   447,   447,
     447,   448,   448,   448,   448,   448,   448,   448,   448,   448,
     448,   448,   449,   450,   450,   450,   451,   451,   452,   452,
     452,   453,   453,   453,   453,   454,   455,   455,   456,   456,
     456,   456,   456,   457,   457,   457,   458,   458,   459,   459,
     459,   460,   460,   461,   461,   462,   462,   463,   464,   465,
     465,   465,   466,   466,   467,   468,   469,   469,   470,   470,
     470,   470,   471,   472,   472,   473,   473,   474,   474,   475,
     476,   476,   478,   477,   477,   479,   479,   479,   479,   480,
     480,   481,   481,   482,   483,   483,   484,   484,   485,   487,
     486,   488,   488,   489,   489,   490,   490,   491,   492,   493,
     494,   494,   495,   495,   496,   496,   496,   497,   497,   498,
     498,   499,   499,   499,   500,   500,   500,   500,   500,   500,
     500,   500,   500,   500,   500,   500,   500,   500,   500,   500,
     500,   500,   500,   500,   500,   500,   500,   500,   500,   501,
     501,   502,   503,   503,   503,   504,   504,   504,   504,   505,
     506,   506,   506,   506,   507,   508,   508,   509,   510,   510,
     511,   511,   511,   511,   511,   512,   512,   512,   512,   513,
     514,   514,   514,   515,   516,   516,   517,   517,   517,   517,
     517,   517,   517,   517,   519,   520,   518,   521,   521,   522,
     522,   522,   523,   523,   523,   524,   525,   525,   526,   526,
     526,   526,   526,   526,   526,   526,   526,   526,   526,   526,
     526,   526,   526,   526,   526,   526,   526,   526,   526,   526,
     526,   526,   527,   527,   528,   528,   528,   529,   529,   530,
     530,   530,   530,   530,   531,   532,   532,   532,   532,   533,
     533,   534,   534,   535,   535,   536,   537,   538,   539,   540,
     540,   541,   542,   542,   543,   544,   544,   545,   545,   546,
     546,   546,   546,   546,   547,   547,   547,   547,   547,   547,
     547,   547,   547,   547,   547,   547,   547,   547,   547,   547,
     547,   547,   548,   549,   549,   549,   549,   550,   550,   551,
     552,   552,   553,   553,   553,   554,   554,   555,   556,   557,
     558,   558,   558,   558,   558,   558,   558,   558,   558,   558,
     558,   558,   558,   558,   559,   560,   560,   560,   561,   562,
     562,   563,   563,   564,   564,   565,   565,   566,   566,   567,
     567,   567,   567,   567,   567,   568,   569,   570,   571,   571,
     572,   572,   573,   573,   574,   574,   574,   575,   575,   575,
     575,   575,   575,   576,   576,   576,   576,   576,   577,   578,
     579,   579,   580,   580,   580,   580,   580,   580,   580,   580,
     580,   581,   581,   582,   582,   582,   582,   582,   582,   582,
     582,   582,   582,   582,   582,   582,   582,   582,   582,   582,
     582,   582,   583,   583,   583,   583,   583,   583,   583,   583,
     583,   583,   583,   583,   583,   583,   583,   583,   583,   583,
     583,   583,   583,   583,   583,   583,   583,   583,   583,   583,
     583,   583,   583,   583,   583,   583,   583,   583,   583,   583,
     583,   583,   583,   583,   583,   583,   584,   584,   585,   586,
     587,   587,   588,   589,   590,   590,   591,   592,   593,   594,
     594,   595,   595,   595,   596,   597,   597,   597,   598,   598,
     599,   600,   600,   601,   602,   602,   603,   604,   604,   605,
     606,   606,   607,   608,   608,   609,   609,   610,   610,   611,
     612,   613,   614,   614,   615,   616,   616,   617,   618,   618,
     618,   618,   618,   618,   618,   618,   619,   620,   621,   621,
     622,   622,   623,   623,   624,   625,   625,   626,   626,   626,
     627,   628,   628,   629,   629,   630,   631,   631,   632,   633,
     633,   634,   634,   635,   636,   637,   638,   639,   639,   640,
     640,   640,   641,   642,   642,   643,   643,   643,   644,   644,
     645,   645,   646,   646,   646,   646,   646,   646,   646,   646,
     646,   646,   646,   647,   649,   648,   648,   650,   650,   651,
     652,   653,   653,   654,   655,   656,   656,   657,   658,   658,
     659,   659,   660,   661,   662,   663,   663,   664,   664,   665,
     666,   666,   667,   667,   668,   668,   669,   669,   670,   670,
     670,   670,   670,   670,   670,   670,   670,   670,   670,   671,
     671,   672,   672,   673,   674,   674,   675,   676,   677,   677,
     677,   678,   678,   679,   680,   680,   681,   681,   682,   683,
     683,   684,   685,   686,   686,   686,   687,   687,   688,   688,
     688,   689,   689,   690,   691,   691,   692,   692,   692,   692,
     692,   692,   692,   693,   694,   695,   696,   697,   697,   697,
     698,   699,   700,   701,   701,   702,   702,   703,   703,   703,
     703,   703,   703,   703,   703,   703,   704,   704,   705,   705,
     705,   705,   705,   706,   707,   707,   708,   708,   708,   708,
     709,   710,   711,   711,   712,   712,   713,   713,   714,   715,
     716,   717,   718,   719,   719,   720,   721,   721,   722,   722,
     723,   723,   724,   724,   725,   725,   726,   727,   727,   727,
     727,   727,   728,   729,   730,   730,   731,   732,   732,   733,
     734,   734,   735,   736,   737,   737,   738,   738,   739,   739,
     740,   740,   740,   740,   741,   742,   743,   744,   745,   746,
     746,   747,   748,   748,   749,   749,   750,   751,   752,   753,
     754,   754,   755,   756,   757,   758,   759,   760,   761,   762,
     762,   763,   763,   763,   764,   765,   766,   766,   767,   767,
     768,   768,   769,   770,   770,   771,   771,   772,   772,   773,
     774,   775,   775,   775,   776,   777,   777,   778,   779,   780,
     780,   781,   782,   783,   783,   783,   783,   783,   783,   783,
     783,   783,   783,   783,   783,   783,   783,   783,   783,   783,
     783,   783,   783,   783,   783,   783,   783,   783,   783,   784,
     785,   786,   786,   787,   787,   788,   788,   788,   788,   788,
     788,   788,   788,   788,   789,   790,   791,   792,   793,   794,
     795,   796,   796,   796,   797,   798,   799,   800,   801,   802,
     802,   802,   802,   802,   802,   802,   802,   802,   802,   802,
     802,   802,   802,   803,   803,   804,   804,   805,   805,   805,
     805,   806,   806,   807,   808,   808,   809,   809,   810,   810,
     811,   811,   811,   811,   811,   811,   812,   813,   813,   814,
     814,   814,   814,   815,   815,   816,   816,   817,   817,   818,
     818,   819,   819,   820,   820,   821,   822,   823,   824,   824,
     825,   825,   826,   826,   827,   827,   827,   827,   827,   827,
     827,   827,   827,   827,   827,   828,   829,   829,   830,   831,
     831,   832,   833,   834,   835,   836,   837,   838,   839,   840,
     840,   841,   841,   842,   842,   843,   844,   844,   844,   844,
     845,   846,   847,   847,   847,   847,   847,   847,   847,   848,
     849,   850,   850,   850,   851,   851,   851,   852,   852,   853,
     853,   854,   854,   854,   855,   855,   855,   855,   855,   856,
     857,   858,   859,   859,   860,   860,   861,   862,   863,   864,
     864,   865,   865,   865,   865,   865,   865,   866,   867,   868,
     869,   869,   869,   870,   870,   871,   872,   872,   873,   873,
     874,   875,   875,   876,   876,   876,   876,   876,   877,   878,
     879,   880,   881,   882,   882,   883,   884,   884,   885,   885,
     885,   886,   887,   888,   889,   889,   890,   890,   890,   890,
     890,   890,   890,   890,   891,   892,   893,   893,   894,   895,
     895,   896,   896,   896,   896,   896,   897,   897,   898,   898,
     899,   899,   900
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     3,     3,     3,     3,     3,     2,     1,
       1,     1,     3,     3,     4,     5,     5,     3,     4,     3,
       0,     2,     2,     2,     1,     1,     4,     5,     4,     5,
       2,     5,     1,     0,     1,     0,     1,     0,     2,     3,
       1,     3,     1,     1,     1,     0,     0,     0,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     4,     2,     5,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     2,     3,     5,
       5,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     4,     7,     2,     3,     7,     6,     0,
       2,     5,     1,     4,     1,     1,     1,     2,     1,     4,
       1,     1,     1,     1,     1,     1,     2,     2,     2,     1,
       1,     7,     3,     4,     3,     4,     3,     2,     5,     0,
       2,     2,     5,     0,     4,     5,     0,     2,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     0,
       1,     5,     6,     6,     0,     1,     5,     9,     1,     1,
       2,     2,     0,     0,     2,     2,     5,     4,     4,     3,
       4,     3,     4,     3,     3,     4,     5,     3,     4,     5,
       3,     3,     1,     3,     2,     4,     3,     4,     3,     3,
       3,     3,     3,     3,     1,     4,     1,     1,     4,     3,
       0,     0,     4,     1,     3,     1,     3,     2,     3,     3,
       4,     2,     0,     1,     1,     3,     5,     1,     3,     0,
       1,     7,     1,     3,     2,     2,     3,     1,     1,     4,
       3,     2,     1,     1,     1,     1,     3,     1,     3,     1,
       1,     6,     1,     1,     2,     2,     1,     3,     1,     2,
       2,     1,     3,     1,     3,     5,     1,     1,     1,     2,
       2,     3,     3,     1,     3,     3,     1,     3,     1,     1,
       3,     1,     3,     1,     1,     3,     5,     0,     0,     1,
       4,     4,     1,     3,     3,     2,     1,     3,     3,     6,
       6,     7,     1,     1,     3,     1,     1,     1,     3,     3,
       0,     3,     0,     2,     3,     1,     1,     2,     3,     1,
       1,     1,     3,     1,     3,     1,     1,     3,     4,     0,
       2,     2,     1,     1,     3,     1,     3,     1,     0,     0,
       0,     2,     0,     1,     1,     1,     2,     2,     4,     1,
       3,     1,     3,     1,     1,     1,     1,     3,     3,     3,
       3,     3,     2,     2,     2,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     2,     3,     3,     1,
       1,     1,     1,     1,     1,     5,     6,     4,     5,     3,
       1,     1,     5,     4,     2,     0,     1,     5,     0,     1,
       1,     3,     1,     3,     2,     1,     1,     1,     1,     1,
       1,     3,     3,     5,     1,     1,     3,     2,     5,     4,
       4,     3,     2,     1,     0,     0,     6,     1,     1,     1,
       4,     5,     1,     4,     5,     0,     1,     3,     1,     1,
       1,     2,     3,     3,     2,     1,     2,     2,     2,     3,
       7,     3,     3,     1,     2,     2,     1,     2,     3,     1,
       1,     1,     5,     7,     0,     6,     4,    11,    13,     4,
       3,     3,     7,     8,     3,     1,     2,     2,     3,     1,
       3,     0,     1,     0,     1,     1,     2,     5,     6,     1,
       3,     3,     0,     2,     1,     5,     7,     0,     1,     3,
       3,     6,     5,     6,     4,     5,     5,     2,     1,     1,
      10,     1,     3,     4,     3,     3,     3,     3,     6,     6,
       5,     8,     2,     3,     3,     7,     7,     0,     1,     4,
       2,     4,     1,     2,     2,     1,     1,     0,     0,     0,
       2,     2,     2,     2,     2,     1,     2,     2,     3,     4,
       2,     3,     1,     3,     3,     1,     1,     1,     3,     1,
       1,     4,     5,     1,     1,     3,     3,     1,     4,     1,
       1,     1,     2,     2,     2,     1,     3,     3,     4,     4,
       1,     3,     1,     5,     1,     1,     1,     3,     3,     3,
       3,     3,     3,     1,     3,     5,     5,     5,     0,     0,
       1,     3,     1,     1,     3,     3,     3,     3,     2,     3,
       3,     0,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     3,     3,     2,     3,
       1,     3,     1,     3,     1,     3,     1,     4,     3,     1,
       3,     1,     3,     4,     1,     4,     4,     4,     3,     3,
       1,     3,     3,     1,     3,     3,     1,     3,     3,     1,
       3,     0,     5,     6,     8,     1,     3,     1,     1,     1,
       4,     1,     0,     2,     3,     2,     4,     0,     1,     5,
       4,     6,     4,     1,     4,     4,     1,     6,     1,     3,
       1,     3,     1,     4,     1,     1,     3,     1,     1,     3,
       1,     0,     1,     2,     3,     1,     2,     5,     4,     4,
       6,     1,     3,     1,     1,     6,     4,     1,     3,     1,
       1,     1,     1,     1,     3,     1,     1,     1,     6,     4,
       1,     4,     1,     1,     1,     1,     4,     2,     7,     1,
       4,     1,     1,    11,     0,     2,     3,     1,     3,     1,
       3,     1,     3,     1,     3,     1,     3,     1,     3,     8,
       1,     3,     2,     2,     7,     1,     3,     3,     1,     4,
       1,     3,     1,     1,     0,     1,     1,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     8,
       6,     8,     6,     1,     6,     6,     6,     6,     1,     3,
       5,     1,     3,     6,     1,     3,     8,     6,     6,     4,
       5,     5,     0,     2,     2,     0,     1,     3,     1,     4,
       7,     1,     3,     3,     1,     3,     5,     3,     3,     1,
       3,     1,     1,     3,     3,     3,     3,    10,     8,    10,
       0,     0,     1,     2,     4,     4,     6,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     6,     4,
       4,     3,     9,     1,     1,     3,     1,     5,     5,     9,
       0,     1,     1,     3,     3,     3,     3,     3,     6,     3,
       3,     3,     3,     7,     5,     1,     1,     3,     4,     1,
       1,     3,     1,     1,     3,     3,     2,     3,     4,     4,
       5,     5,     1,     2,     4,     4,     4,     0,     1,     2,
       7,     6,     3,     3,     7,     5,     1,     3,     1,     4,
       2,     3,     3,     4,     6,     3,     2,     3,     1,     1,
       4,     5,     3,     6,     2,     4,     2,     1,     3,     3,
       0,     1,     3,     2,     2,     2,     2,     9,     5,     1,
       3,     2,     2,     2,     9,     4,     1,     3,     1,     1,
       2,     0,     7,     1,     4,     1,     3,     1,     1,     1,
      16,     0,     3,     3,     3,     3,     6,     9,     5,     1,
       3,     5,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       2,     4,     3,     4,     5,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     5,     2,     2,     2,     2,     2,
       5,     1,     1,     1,     4,     4,     4,     4,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     4,     3,     4,     5,     1,     1,     1,
       1,     4,     3,     2,     4,     3,     4,     3,     4,     5,
       1,     1,     1,     1,     1,     1,     1,     7,     5,     1,
       1,     1,     1,     4,     3,     4,     5,     1,     1,     4,
       3,     4,     5,     1,     1,     2,     1,     2,     4,     3,
       4,     3,     4,     5,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     2,     4,     3,     2,     4,
       3,     2,     3,     2,     2,     2,     2,     2,     2,     3,
       2,     5,     2,     5,     2,     5,     1,     1,     3,     3,
       0,     0,     1,     1,     1,     1,     1,     1,     1,     3,
       2,     5,     4,     2,     5,     4,     2,     2,     1,     1,
       3,     2,     2,     2,     4,     4,     4,     4,     4,     4,
       1,     1,     1,     3,     2,     2,     1,     1,     3,     2,
       2,     1,     1,     1,     1,     1,     1,     5,     5,     5,
       3,    10,    10,     1,     3,     2,     0,     6,     0,     6,
       2,     1,     3,     1,     1,     1,     1,     1,     5,     5,
       5,     5,     5,     1,     3,     3,     1,     3,     1,     1,
       1,     5,     5,     5,     1,     3,     2,     5,     2,     5,
       5,     2,     5,     2,     5,     1,     1,     3,     5,     1,
       3,     5,     5,     5,     5,     7,     1,     3,     2,     2,
       2,     2,     0
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       2,     0,     1,    10,    11,     9,     0,     0,     3,   150,
     149,   789,   339,   537,   537,   537,   537,   537,   339,   537,
     555,   537,    20,   537,   537,   158,   537,   560,   339,   154,
     537,   339,   537,   537,   339,   485,   537,   537,   537,   338,
     537,   782,   537,   537,   340,   791,   537,   155,   156,   785,
      45,   537,   537,   537,   537,   537,   537,   537,   537,   557,
     537,   537,   483,   483,   537,   537,   537,   339,   537,     0,
     537,   339,   339,   537,   537,   338,    20,   339,   339,   325,
       0,   537,   537,   339,   339,   537,   339,   339,   339,   339,
     152,   339,   537,   211,   537,   157,   537,   537,     0,    20,
     339,   537,   537,   537,   559,   339,   537,   339,   535,   537,
     537,   339,   537,   537,   537,   339,    20,   339,    45,   537,
     537,   153,    45,   537,   339,   537,   537,   537,   339,   537,
     537,   556,   339,   537,   339,   537,   537,   537,   537,   537,
     537,   339,   339,   536,    20,   339,   339,   537,   537,   537,
       0,   339,     8,   339,   537,   537,   537,   783,   537,   537,
     537,   537,   537,   537,   537,   537,   537,   537,   537,   537,
     537,   537,   537,   537,   537,   537,   537,   537,   537,   537,
     537,   537,   537,   537,   339,   537,   784,   537,  1226,   537,
    1227,   537,   537,   339,   537,   537,   537,   537,   537,   537,
     537,   537,  1302,  1302,  1302,  1302,  1302,  1302,   611,     0,
      37,   611,    73,    48,    49,    50,    65,    66,    76,    68,
      69,    67,   109,    58,     0,   146,   151,    52,    70,    71,
      72,    51,    59,    54,    55,    56,    60,   207,    75,    74,
      57,   611,   445,   440,   453,     0,     0,     0,   456,   439,
     438,     0,   508,   511,   537,   509,     0,   537,     0,   537,
     545,     0,     0,   552,    53,   459,   613,   616,   622,   618,
     617,   623,   624,   625,   626,   615,   632,   614,   633,   619,
       0,   780,   620,   627,   629,   628,   660,   634,   636,   637,
     635,   638,   639,   640,   641,   642,   621,   643,   644,   646,
     647,   645,   649,   650,   648,   674,   661,   662,   663,   664,
     651,   652,   653,   655,   654,   656,   657,   658,   659,   665,
     666,   667,   668,   669,   670,   671,   672,   673,   631,   675,
     630,  1034,  1033,  1035,  1036,  1037,  1038,  1039,  1040,  1041,
    1042,  1043,  1044,  1045,  1046,  1047,  1048,  1049,  1032,  1050,
    1051,  1052,  1053,  1054,  1055,  1056,  1057,  1058,   460,  1192,
    1194,  1196,  1197,  1193,  1195,  1198,   461,  1231,  1232,  1233,
    1234,  1235,  1236,     0,     0,   340,     0,     0,     0,     0,
       0,     0,     0,   996,    35,     0,     0,   598,     0,     0,
       0,     0,     0,     0,   454,   507,   481,   210,     0,     0,
       0,   481,     0,   312,   339,   727,     0,   727,   538,     0,
      23,   481,     0,   481,   976,     0,   993,   483,   481,   481,
     481,    32,   484,    81,   444,   959,   481,   953,   105,   481,
      37,   481,     0,   340,     0,     0,    63,     0,     0,   329,
      44,     7,   970,     0,     0,     0,   599,     0,     0,    77,
     340,     0,   990,   522,     0,     0,     0,   296,   295,     0,
       0,   813,     0,     0,   340,     0,     0,   538,     0,   340,
       0,     0,     0,   340,    33,   340,    22,   599,     0,    21,
       0,     0,     0,     0,     0,     0,     0,   398,   340,    45,
     140,     0,     0,     0,     0,     0,     0,     0,     0,   787,
     340,     0,   340,     0,     0,   994,   995,     0,   339,   340,
       0,     0,     0,   599,     0,  1178,  1177,  1182,  1059,   727,
    1184,   727,  1174,  1176,  1060,  1165,  1168,  1171,   727,   727,
     727,  1180,  1173,  1175,   727,   727,   727,   727,  1113,   727,
     727,  1190,  1147,     0,    45,  1200,  1203,  1206,    45,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,  1250,     0,   612,     4,    20,    20,     0,     0,    45,
       5,     0,     0,     0,     0,     0,    45,    20,     0,     0,
       0,   147,   164,     0,     0,     0,     0,   528,     0,   528,
       0,     0,     0,     0,   528,   222,     6,   486,   537,   537,
     446,   441,     0,   457,   448,   447,   455,    82,   172,     0,
       0,     0,   406,     0,   405,   410,   408,   409,   407,   381,
       0,     0,   351,   382,   354,   384,   383,   355,   400,   402,
     395,   353,   356,   598,   398,   542,   543,     0,   380,   379,
      32,     0,   602,   603,   540,     0,   600,   599,     0,   544,
     599,   564,   547,   546,   600,   550,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    45,     0,   776,   777,   775,
       0,   773,   763,     0,     0,   435,     0,   323,     0,   524,
     978,   979,   975,    45,   310,   808,   810,   977,    36,    13,
     598,     0,   481,     0,   192,     0,   310,     0,   184,     0,
     709,   707,   843,   931,   932,   807,   804,   805,   482,   516,
     222,   435,   310,   676,   987,   982,   470,   341,     0,     0,
       0,     0,     0,   719,   722,   711,     0,   497,   682,   679,
     680,   451,     0,     0,   500,   988,   442,   443,   458,   452,
     471,   106,   499,    45,   517,     0,   199,     0,   382,     0,
       0,    37,    25,   803,   800,   801,   324,   326,     0,     0,
      45,   971,   972,     0,   700,   698,   686,   683,   684,     0,
       0,     0,    78,     0,    45,   991,   989,     0,     0,   952,
       0,    45,     0,    19,     0,     0,     0,     0,   957,     0,
       0,     0,   497,   523,     0,     0,   935,   962,   599,     0,
     599,   600,   139,    34,    12,   143,   576,     0,   764,     0,
       0,     0,   727,   706,   704,   892,   929,   930,     0,   703,
     701,   963,   399,   514,     0,     0,     0,   913,     0,   925,
     924,   927,   926,     0,   691,     0,   689,   694,     0,     0,
      37,    24,     0,   310,   944,   947,     0,    45,     0,   302,
     298,     0,     0,   577,   310,     0,   527,     0,  1117,  1112,
     527,  1149,  1179,     0,   527,   527,   527,   527,   527,   527,
    1172,   310,    46,  1199,  1208,  1209,     0,     0,    46,  1228,
      45,     0,  1024,  1025,     0,   992,   349,     0,     0,    45,
      45,    45,  1285,  1240,    45,     0,     0,    20,    43,    38,
      42,     0,    40,    17,    46,   310,   132,   134,   136,   110,
       0,     0,    20,   339,   148,   538,   598,   165,   146,   310,
     179,   181,   183,   187,   527,   190,   527,   196,   198,   200,
     209,     0,   213,     0,    45,     0,   449,   424,     0,   351,
     364,   363,   376,   362,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   598,     0,     0,   598,     0,     0,   398,
     404,   396,   512,     0,     0,   515,   570,   571,   575,     0,
     567,     0,   569,     0,   608,     0,     0,     0,     0,     0,
     554,   569,   558,     0,     0,   582,   548,   580,     0,     0,
     351,   353,   551,   586,   585,   553,   677,   310,   699,   702,
     705,   708,   310,   339,     0,   945,     0,    45,   758,   178,
       0,     0,     0,     0,     0,     0,   312,   812,     0,   529,
       0,   475,   479,     0,   469,   598,     0,   194,   185,     0,
     321,     0,   208,     0,   678,   598,     0,   786,   319,   316,
     313,   315,   320,   310,   727,   724,   733,   728,     0,     0,
       0,     0,     0,   725,   711,   727,     0,   790,     0,   498,
     539,     0,     0,     0,    18,   204,     0,     0,     0,   206,
     195,     0,   494,   492,   489,     0,    45,     0,   329,     0,
       0,   332,   330,     0,    45,   973,   381,   921,   968,     0,
       0,   966,     0,   561,     0,    87,    88,    86,    85,    91,
      90,   102,    95,    98,    97,   100,    99,    96,   101,    94,
      92,    89,    93,    83,     0,    84,   197,     0,     0,     0,
       0,     0,     0,   297,     0,   188,   436,     0,    45,   958,
     956,   133,   815,     0,     0,     0,   292,   539,   180,     0,
     579,     0,   578,   287,   287,     0,   759,     0,   727,   711,
     939,     0,     0,   936,   284,   283,    62,   281,     0,     0,
       0,     0,     0,     0,     0,   688,   687,   135,    14,   182,
     946,    45,   949,   948,   146,     0,   103,    47,     0,     0,
     695,     0,   727,   527,     0,  1146,  1116,  1111,   727,   527,
    1148,  1191,   727,   527,   727,   527,   527,   527,   727,   527,
     727,   527,   696,     0,     0,     0,     0,  1220,     0,     0,
    1207,  1211,  1213,  1212,    45,  1202,   851,  1221,     0,  1205,
       0,  1229,  1230,     0,     0,   999,    45,    45,    45,     0,
     390,   391,  1029,     0,     0,     0,     0,  1251,  1253,  1254,
    1255,  1257,  1256,     0,     0,  1266,  1268,  1269,  1270,     0,
       0,  1274,    45,     0,     0,  1289,    28,    37,     0,     0,
      39,     0,    30,   159,   116,   310,   339,   118,   120,     0,
     121,   114,   122,   130,   129,   123,   124,   125,     0,   112,
     115,    26,     0,   310,     0,     0,   144,   177,     0,     0,
     222,   222,     0,   224,   217,   221,     0,     0,     0,   352,
       0,   359,   361,   358,   357,   375,   377,   371,   365,   504,
     368,   366,   369,   367,   370,   372,   374,   360,   373,   378,
     598,   411,   389,     0,   343,     0,   414,   415,   401,   412,
     403,     0,   598,   513,     0,   532,   530,     0,   598,   566,
     573,   574,   572,   601,   610,   605,   607,   609,   606,   604,
     565,   549,     0,     0,     0,   351,     0,     0,     0,     0,
       0,   697,   779,     0,   789,   792,   782,     0,   791,   785,
       0,   783,   784,   781,   774,     0,   429,     0,     0,   506,
       0,     0,     0,     0,   811,   477,   476,     0,   474,     0,
     193,     0,   527,   806,   427,   428,   432,     0,     0,     0,
     314,   317,   176,     0,   598,     0,     0,     0,     0,     0,
     712,   723,   310,   462,   727,   681,     0,   481,     0,     0,
     201,     0,   394,   981,     0,     0,     0,    16,   802,   327,
     337,     0,   333,   335,   331,     0,     0,     0,     0,     0,
       0,     0,   965,   685,   562,    80,    79,   128,   126,   127,
     340,     0,   487,   423,     0,     0,     0,     0,   191,     0,
     520,     0,     0,   727,     0,     0,    64,   527,   505,   601,
     138,     0,   142,    45,     0,   711,     0,     0,     0,     0,
     934,     0,     0,     0,     0,     0,   914,   916,     0,   692,
     690,     0,    45,   951,    45,   950,   145,   340,     0,   502,
       0,  1181,     0,   727,  1183,     0,   727,     0,     0,   727,
       0,   727,     0,   727,     0,   727,     0,     0,     0,    45,
       0,     0,     0,  1210,     0,  1201,  1204,  1003,  1001,  1002,
      45,   998,     0,     0,     0,   350,   598,   598,     0,  1028,
    1031,     0,     0,     0,     0,     0,    45,  1237,     0,     0,
       0,    45,  1238,  1276,  1278,     0,     0,  1281,  1283,    45,
    1239,     0,     0,     0,     0,     0,     0,    45,  1288,    15,
      29,    41,     0,   173,   160,   117,     0,    45,     0,    45,
      27,   159,   539,   539,   169,   172,   168,     0,   186,   189,
     214,     0,     0,     0,   247,   245,   252,   249,   263,   256,
     261,     0,     0,   215,   238,   250,   242,   253,   243,   258,
     244,     0,   237,     0,   232,   229,   218,   219,     0,     0,
     425,   351,     0,   387,   598,   347,   344,   345,     0,   398,
       0,   534,   533,     0,     0,   581,   352,     0,     0,     0,
     351,   588,   351,   592,   351,   590,   310,     0,   598,   518,
       0,     0,   980,     0,   311,   478,   480,   172,   322,     0,
     598,   519,     0,   984,   598,   983,   318,   320,   726,     0,
       0,     0,   736,     0,     0,     0,     0,   710,   464,   481,
     501,     0,   203,   202,   381,   493,   490,   488,     0,   491,
       0,   328,     0,     0,     0,     0,     0,     0,   967,     0,
    1013,     0,     0,   422,   417,   954,   955,   721,   310,   961,
     437,     0,   816,   818,   824,   294,   293,     0,   287,     0,
       0,   289,   288,     0,   760,   761,   713,     0,   943,   942,
       0,   940,     0,   937,   282,     0,  1019,  1008,     0,  1006,
    1009,   755,     0,     0,   928,   920,   693,     0,     0,     0,
       0,     0,   300,     0,   299,   307,     0,  1190,     0,  1190,
    1190,  1126,     0,  1120,  1122,  1123,  1121,   727,  1125,  1124,
       0,  1190,   727,  1144,  1143,     0,     0,  1187,  1186,     0,
       0,  1190,     0,  1190,     0,   727,  1065,  1069,  1070,  1071,
    1067,  1068,  1072,  1073,  1066,     0,  1154,  1158,  1159,  1160,
    1156,  1157,  1161,  1162,  1155,  1164,  1163,   727,     0,  1107,
    1109,  1110,  1108,   727,     0,  1137,  1138,   727,     0,     0,
       0,     0,     0,     0,  1222,     0,     0,   852,  1000,     0,
    1026,     0,   598,     0,  1030,     0,     0,    45,     0,     0,
    1252,     0,     0,     0,  1267,     0,     0,     0,     0,  1275,
       0,     0,    45,     0,     0,     0,    45,  1290,     0,     0,
       0,   108,   794,     0,   111,     0,   173,     0,   146,     0,
     171,   170,   267,   253,   266,     0,   255,   260,   254,   259,
       0,     0,     0,     0,     0,   222,   212,   223,   241,     0,
     222,   234,   235,     0,     0,     0,     0,   278,   223,   279,
       0,     0,   227,   268,   273,   276,   229,   220,     0,   503,
       0,   413,   385,   388,     0,   346,     0,   531,   568,   569,
       0,     0,   351,     0,     0,     0,   778,   772,   788,     0,
       0,     0,   525,     0,   340,   526,     0,   986,     0,     0,
       0,   740,     0,   738,   735,   730,   734,   732,     0,    45,
       0,   463,   450,   205,   334,   336,     0,     0,     0,   969,
     964,   131,     0,  1012,   421,     0,     0,   416,   960,     0,
       0,    45,   814,   825,   826,   831,   835,   828,   836,   837,
     838,   832,   834,   833,   829,   830,     0,     0,     0,     0,
     285,     0,     0,     0,     0,   938,   933,   472,     0,  1005,
     727,   915,     0,     0,   890,   104,   306,   301,   303,   305,
       0,     0,     0,  1075,   727,  1076,  1077,    45,  1118,   727,
    1145,  1141,   727,  1190,     0,  1074,    45,  1078,     0,  1079,
       0,  1063,   727,  1152,   727,  1105,   727,  1135,   727,  1214,
    1215,  1216,  1224,  1225,    45,  1219,  1217,  1218,     0,     0,
       0,   393,     0,     0,  1263,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,  1286,     0,     0,     0,    45,
      45,     0,     0,  1296,     0,     0,     0,     0,     0,    31,
     175,   174,     0,     0,   119,   113,   107,     0,     0,   161,
     598,   166,     0,   248,   246,   264,   257,   262,   216,   222,
     598,     0,   240,   236,   223,     0,   233,     0,   270,   269,
       0,   225,   229,     0,     0,     0,     0,     0,   230,     0,
     426,   386,   348,   397,     0,   583,   595,   597,   596,     0,
     430,     0,     0,   809,     0,   433,     0,   985,   756,   729,
       0,     0,    45,     0,     0,     0,   844,   974,   845,  1018,
       0,  1015,  1017,   420,   419,     0,     0,     0,   817,     0,
     827,     0,   288,     0,     0,   765,   762,   719,   714,   715,
     717,   718,   941,  1007,  1011,     0,     0,   381,     0,     0,
       0,     0,   309,   308,   521,     0,     0,     0,  1119,  1142,
       0,  1189,  1188,     0,     0,     0,  1064,  1153,  1106,  1136,
    1223,     0,     0,   392,     0,     0,  1262,  1259,   902,   903,
     904,   901,   906,   900,   907,   899,   898,   897,   905,   893,
       0,     0,    45,  1258,  1261,  1260,  1272,  1273,  1271,  1279,
    1277,     0,  1280,     0,  1282,     0,     0,  1243,     0,  1298,
    1299,    45,  1291,  1292,  1293,  1294,  1300,  1301,     0,     0,
       0,   795,   162,   163,     0,     0,   239,   598,   241,     0,
     280,   228,     0,   272,   271,   274,   275,   277,   473,     0,
     770,   769,   771,     0,   767,   431,     0,   997,   434,     0,
     741,   739,     0,   731,     0,     0,     0,  1014,   418,   846,
       0,     0,     0,     0,   911,     0,     0,     0,     0,     0,
       0,     0,   286,   291,   290,     0,     0,     0,  1004,   917,
     918,     0,   842,   891,   891,   304,  1091,  1090,  1089,  1096,
    1097,  1098,  1095,  1092,  1094,  1093,  1102,  1099,  1100,  1101,
       0,  1086,  1130,  1129,  1131,  1132,     0,  1191,  1081,  1083,
    1082,     0,  1085,  1084,     0,  1027,  1265,  1264,     0,     0,
       0,  1287,     0,  1245,    45,  1246,  1248,  1297,     0,   796,
       0,   172,   265,     0,     0,   227,   226,     0,     0,   766,
     510,     0,     0,     0,   466,  1016,   823,   822,     0,   820,
     862,   859,     0,     0,     0,     0,   909,   910,     0,     0,
       0,     0,     0,   716,   922,  1010,    45,     0,     0,     0,
       0,     0,  1128,  1185,  1080,    45,     0,     0,   894,     0,
    1244,    45,  1241,    45,  1242,  1295,     0,     0,   251,   231,
     495,   768,   757,   744,   737,   742,     0,     0,   819,   865,
     860,     0,     0,     0,     0,     0,     0,     0,   848,     0,
     854,     0,   467,   720,     0,     0,   841,    45,    45,   888,
    1088,  1087,     0,     0,   895,     0,  1284,     0,     0,   799,
     793,   797,   167,     0,     0,   465,   821,     0,     0,     0,
       0,   857,     0,   840,     0,   908,   858,     0,   847,     0,
     853,     0,   923,     0,     0,     0,  1127,     0,     0,   354,
       0,     0,     0,   496,     0,   747,     0,   745,   748,   863,
     864,     0,   866,   868,     0,     0,     0,   849,   855,   468,
     919,   889,   887,     0,   896,    45,    45,   798,   750,   751,
       0,   743,     0,   861,     0,   856,   839,     0,     0,     0,
       0,     0,     0,   749,   752,   746,   867,     0,     0,   871,
     912,   850,  1021,  1247,  1249,   753,     0,     0,     0,   869,
      45,  1020,   754,   873,   872,    45,     0,     0,     0,   874,
     879,   881,   882,  1022,  1023,     0,     0,     0,    45,   870,
      45,    45,   598,   885,   884,   883,   875,     0,   877,   878,
       0,   880,     0,    45,   886,   876
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     6,     7,   208,   384,   209,   840,   751,   210,
     903,   619,   804,   689,   569,   901,   902,   441,  2245,  1220,
    1508,   211,   212,   620,  1124,  1125,   213,   214,   215,   579,
    1288,  1289,  1128,  1290,   216,   217,   218,   219,  1153,   220,
     221,  1154,   222,   582,   223,   224,   225,   226,  1583,  1584,
     918,  1595,   937,  1871,   227,   228,   229,   230,   231,   232,
     785,  1164,  1165,   233,   234,   235,   746,  1076,  1077,   236,
     237,   710,   453,   930,   931,  1611,   932,   933,  1909,  1621,
    1626,  1627,  1910,  1911,  1622,  1623,  1624,  1613,  1614,  1615,
    1616,  1883,  1618,  1619,  1620,  1885,  2128,  1913,  1914,  1915,
    1166,  1167,  1480,  1481,  2000,  1732,  1145,  1146,   238,   458,
     239,   850,  2017,  2018,  1764,  2019,  1027,   718,   719,  1050,
    1051,  1039,  1040,   240,   756,   757,   758,   759,  1092,  1441,
    1442,  1443,   397,   374,   404,  1333,  1635,  1334,   885,   999,
     622,   641,   623,   624,   625,   626,  2062,  1079,   970,  1923,
     823,   627,   628,   629,   630,   631,  1338,  1637,   632,  1308,
    1920,  1406,  1387,  1407,  1020,  1137,   241,   242,  1961,   243,
     244,   692,  1032,  1033,   709,   423,   245,   246,   247,   248,
    1083,  1084,  1435,  1930,  1931,  1070,   249,   250,   251,   252,
    1202,   253,   973,  1346,   254,   376,   727,  1424,   255,   256,
     257,   258,   259,   260,   652,   644,   979,   980,   981,   261,
     262,   263,   996,   997,  1002,  1003,  1004,  1335,   769,   645,
     801,   564,   264,   265,   266,   713,   267,   729,   730,   268,
     767,   768,   269,   499,   835,   836,   838,   270,   271,   765,
     272,   820,   273,   814,   274,   701,  1067,   275,   276,  2178,
    2179,  2180,  2181,  1718,  1064,   407,   721,   722,  1063,  1683,
    1747,  1952,  1953,  2434,  2435,  2506,  2507,  2529,  2543,  2544,
    1752,  1950,   277,   278,  1734,   673,   809,   810,  1938,  2283,
    2284,  1939,   670,   671,   279,   280,   281,   282,  2092,  2093,
    2470,  2471,   283,   754,   755,   284,   706,   707,   285,   685,
     686,   286,   287,  1143,  1724,  2168,  2388,  2389,  1982,  1983,
    1984,  1985,  1986,   703,  1987,  1988,  1989,  2449,  1227,  1990,
    2451,  1991,  1992,  1993,  2391,  2439,  2479,  2511,  2512,  2548,
    2549,  2568,  2569,  2570,  2571,  2572,  2583,  1994,  2190,  2408,
     816,  2067,  2229,  2230,  2231,  1995,   828,  1495,  1496,  2012,
    1160,  2405,   288,   289,   290,   291,   292,   293,   294,   295,
     797,  1162,  1163,  1740,  1741,   296,   844,   297,   780,   298,
     781,   299,  1140,   300,   301,   302,   303,   304,  1100,  1101,
     305,   762,   306,   307,   308,   681,   682,   309,   310,  1409,
    1673,   715,   311,   312,   776,   313,   314,   315,   316,   317,
     318,   319,  1234,  1235,   320,  1170,  1748,  1749,  2318,   321,
    1711,  2160,  2161,  1750,   322,  2561,   323,   324,   325,   326,
    1243,   327,   328,   329,   330,   331,   332,  1203,  1795,   862,
    1773,  1774,  1775,  1799,  1800,  1801,  2351,  1802,  1803,  1776,
    2196,  2461,  2340,   333,  1209,  1823,   334,   335,   336,   337,
    1193,  1777,  1778,  1779,  2346,   338,  1211,  1827,   339,  1199,
    1782,  1783,  1784,   340,   341,   342,  1205,  1817,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,  1788,  1789,   863,  1517,   358,   359,   360,
     361,   362,   873,   874,   875,  1221,  1222,  1223,  1228,  1833,
    1834,   363,   364,   365,   879,   366,   367,   368,   369,   370,
    2246,  2247,  2422,  2424,   371,  1246,  1247,  1248,  1249,  1250,
    1251,  1252,  2063,  2064,  1254,  1255,  1256,  1257,  1258,  1260,
    1261,  2078,   893,  2076,   372,  1264,  1265,  2082,  2083,  2088,
     557
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -2238
static const yytype_int16 yypact[] =
{
   -2238,   121, -2238, -2238, -2238, -2238,   108,  5149, -2238, -2238,
   -2238,   124, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,   904, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238,   130, -2238, -2238,   516,   126, -2238, -2238, -2238,   130,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238,   199,   199, -2238, -2238, -2238, -2238, -2238,   199,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
     186, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,   199, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238,   279, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
     316,   390, -2238, -2238, -2238, -2238, -2238,   130, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238,   130, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,   189,   991,
     438,   189, -2238, -2238, -2238,   462,   563,   569,   610, -2238,
   -2238, -2238,   521,   665,   199, -2238, -2238,   695,   711,   725,
     732,   638,   188,   735,   771,   792, -2238,   568, -2238, -2238,
   -2238,   189, -2238, -2238, -2238,   631,   744,  2244,  2336, -2238,
   -2238,  2957, -2238,   811, -2238, -2238,  1868, -2238,   847, -2238,
   -2238,  1643,   847,   863, -2238, -2238,   897, -2238, -2238, -2238,
     912,   919,   937,   955,   964, -2238, -2238, -2238, -2238,  1011,
    1079, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238,  1023, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238,   173,   199,  1002,  1028,  1046,   934,   199,
     199,   117,   199, -2238,   199,   199,  1053, -2238,   621,  1064,
     199,   199,   199,   199, -2238, -2238,   199, -2238,  1080,   199,
     948,   199,  1137, -2238, -2238, -2238,   199, -2238,  1101,   199,
   -2238,   199,  1103,   140, -2238,   948, -2238,   199,   199,   199,
     199, -2238, -2238, -2238, -2238, -2238,   199, -2238,   199,   199,
     438,   199,  1154,  1002,   199,  1156, -2238,   199,   199, -2238,
   -2238, -2238,  1147,  1178,   199,   199, -2238,  1187,  1222,   199,
    1002,  1227,  2957, -2238,  1232,  1238,   199, -2238,  1203,   199,
    1246, -2238,  1244,   199,  1002,  1262,  1270, -2238,   934,  1002,
     199,   199,  1931,    66,   199,   110, -2238, -2238,   166, -2238,
     168,   199,   199,   199,  1276,   199,   199,  2957,   119, -2238,
   -2238,  1282,   199,   199,   199,   199,   199,  2702,   199, -2238,
    1002,   199,  1002,   199,   199, -2238, -2238,   199, -2238,  1002,
     199,  1297,  1299, -2238,   199, -2238, -2238,  1315, -2238, -2238,
    1325, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238,  1334, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238,   199, -2238, -2238,  1365,  1368, -2238,  1383,
    2957,  2957,  2957,  2957,  2957,  1387,  1389,  1394,  1398,  1420,
     199, -2238,  1426, -2238, -2238, -2238, -2238,  1194,   193, -2238,
   -2238,   199,   199,   199,   199,  1431, -2238, -2238,  1326,   199,
     199, -2238,   243,   199,   199,   199,   199,   199,   559,   199,
    1246,   199,   199,  1154, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238,  1230, -2238, -2238, -2238, -2238, -2238, -2238,  2957,
    2957,  2957, -2238,  2957, -2238, -2238, -2238, -2238, -2238, -2238,
    2957,  1328, -2238,   221,  1285, -2238,  1430, -2238,  1220,  1239,
    1446, -2238, -2238,  1452,  2957, -2238, -2238,  1580, -2238, -2238,
    1466,  1472,  1285, -2238, -2238,   812,    -3, -2238,  1580, -2238,
   -2238, -2238,  1496,   228,   144,  2988,  2988,   199,   199,   199,
     199,   199,   199,   199,  1503, -2238,   199, -2238, -2238, -2238,
     643, -2238, -2238,  1492,   199, -2238,  2957, -2238,  1272,   179,
   -2238,  1506, -2238, -2238,  1519,  1510, -2238, -2238, -2238, -2238,
   -2238,  2753,   199,  1523, -2238,   199,  1519,   199, -2238,   934,
   -2238, -2238, -2238, -2238, -2238, -2238,  1537, -2238, -2238, -2238,
   -2238, -2238,  1519, -2238, -2238,  1531, -2238, -2238,   740,  1441,
     199,   749,   154, -2238,  1532,  1377,  2957,  1402, -2238,  1541,
   -2238, -2238,  2957,  2957, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238,   199, -2238,   199,  1538,   165,
     199,   438, -2238, -2238,  1547, -2238,  1548, -2238,  1542,   826,
   -2238,  1550, -2238,   199, -2238, -2238, -2238,  1552, -2238,   847,
    1535,  3122, -2238,   199, -2238,  5826, -2238,   199,  2957, -2238,
    1551, -2238,   199, -2238,   199,   199,   199,  1285,   714,   199,
     199,   199,  1402, -2238,   199,   718, -2238, -2238, -2238,  1472,
     812, -2238, -2238, -2238, -2238, -2238, -2238,   173, -2238,  1492,
    1555,  1532, -2238, -2238, -2238, -2238, -2238, -2238,   199, -2238,
   -2238, -2238,  5826, -2238,   621,  1501,   199, -2238,  1554, -2238,
   -2238, -2238, -2238,  1556,  3194,   763, -2238, -2238,   279,   199,
     438, -2238,   199,  1519, -2238,  1559,  1558, -2238,   199, -2238,
    1565,  2957,  2957, -2238,  1519,   199,   177,   199,  1288,  1288,
     184,  1288, -2238,  1561,   202,   222,   229,   348,   352,   386,
   -2238,  1519,   582, -2238,  1569, -2238,   156,   196,  1267, -2238,
   -2238,  3225,  5826,  3260,  3304,  1574,  5826,   199,   199, -2238,
   -2238, -2238, -2238,  1575, -2238,   199,   199, -2238, -2238, -2238,
   -2238,   790, -2238, -2238,  1371,  1519, -2238, -2238, -2238, -2238,
    2845,   199, -2238, -2238, -2238, -2238, -2238, -2238, -2238,  1519,
   -2238, -2238, -2238, -2238,  1583, -2238,  1583, -2238, -2238, -2238,
   -2238,   481, -2238,   437, -2238,  1573, -2238, -2238,  3456,  1584,
    1589,  1589,  1897, -2238,  2957,  2957,  2957,  2957,  2957,  2957,
    2957,  2957,  2957,  2957,  2957,  2957,  2957,  2957,  2957,  2957,
    2957,  2957,  2957, -2238,  1529,  1470,  1581,   388,    98,  2957,
   -2238, -2238, -2238,   795,  1486, -2238, -2238, -2238, -2238,   798,
   -2238,  1825,   865,  2957,  1594,  1472,  1472,  1472,  1472,  1472,
   -2238,   909, -2238,   228,   228,  1285,  1597, -2238,  2988,  5826,
     115,   152, -2238,  1600,  1605, -2238, -2238,  1519, -2238, -2238,
   -2238, -2238,  1519, -2238,   604, -2238,   173, -2238, -2238, -2238,
     199,  3579,   199,  1599,  2957,  1553, -2238, -2238,   199, -2238,
    2957,  3714, -2238,   806, -2238, -2238,  1585, -2238, -2238,   829,
   -2238,   199, -2238,   199, -2238, -2238,  1441, -2238, -2238, -2238,
   -2238, -2238,  3808,  1519, -2238, -2238, -2238,  1602,  1603,  1608,
    1611,  1612,  1614, -2238,  1377, -2238,   199, -2238,  3842, -2238,
   -2238,   199,  3876,  3910, -2238,  1616,   835,  1624,  1446, -2238,
   -2238,   199, -2238,  1625, -2238,  1623, -2238,   199, -2238,  1505,
     669, -2238, -2238,    21, -2238, -2238,  1646, -2238,  1635,  1647,
     840, -2238,   199,  2988,  1634, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238,  1641, -2238, -2238,   454,  1642,  1638,
    3986,  2799,   -57, -2238,  1627, -2238, -2238,   864, -2238, -2238,
   -2238, -2238, -2238,   885,  1639,   900, -2238, -2238, -2238,  2957,
   -2238,  1001, -2238, -2238, -2238,   901, -2238,  1659, -2238,  1377,
    1652,  1661,   902, -2238, -2238, -2238,  1667, -2238,  1654,  1662,
    1650,   199,  2957,  2957,  2702, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238,  1672,  1673, -2238,   -37, -2238, -2238,  4021,  4056,
   -2238,  1663, -2238,   423,  1674, -2238, -2238, -2238, -2238,   463,
   -2238, -2238, -2238,   484, -2238,   488,   492,   552, -2238,   602,
   -2238,   618, -2238,  1668,  1675,  1678,  1681, -2238,  1687,  1688,
   -2238, -2238, -2238, -2238, -2238, -2238,  1285,  1680,  1695, -2238,
    1696, -2238, -2238,   -29,   907, -2238, -2238, -2238, -2238,  2957,
     517,   723, -2238,   926,   928,    93,   930, -2238, -2238, -2238,
   -2238, -2238, -2238,   120,   945, -2238, -2238, -2238, -2238,   549,
     952, -2238, -2238,   508,   953, -2238, -2238,   438,   199,   145,
   -2238,  1693, -2238,  1677, -2238,  1519, -2238, -2238, -2238,  1703,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,  1369, -2238,
   -2238, -2238,   199,  1519,   116,  1974, -2238, -2238,   199,   199,
   -2238,  1998,   437, -2238,  1705, -2238,  1660,  2957,  2988, -2238,
    2957,  1589,  1589,   630,   630,  1897,  1075,  3364,  2634,  5826,
    2634,  2634,  2634,  2634,  2634,  3364,  1669,  1589,  1669,  5857,
    1581, -2238, -2238,  1700,  1718,  2409, -2238, -2238, -2238, -2238,
   -2238,  1720, -2238, -2238,   934,  5826, -2238,  2957, -2238, -2238,
   -2238, -2238,  5826,    73,  5826,  1594,  1594,  1161,  1594,   644,
   -2238,  1597,  1724,   228,  4091,  1725,  1729,  1730,  2988,  2988,
    2988, -2238, -2238,   199,  1713, -2238, -2238,  1726,  1532, -2238,
     279, -2238, -2238, -2238, -2238,  1484, -2238,   984,   934, -2238,
     934,   987,  1733,   990, -2238,  5826,  2957,  2753, -2238,   999,
   -2238,   934,  1583, -2238,   793,   916, -2238,  1006,  1576,  1008,
   -2238,  2202, -2238,   154, -2238,  1732,   199,   199,  2957,   199,
   -2238, -2238,  1519, -2238, -2238, -2238,  1511,   199,  2957,   199,
   -2238,   199, -2238,  1285,  2957,  1731,  2799, -2238, -2238, -2238,
   -2238,  1031, -2238,  1735, -2238,  1739,  1748,  1749,  1557,  2957,
     199,   199, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
    1002,   199, -2238,  2832,  3091,  1745,   199,   199, -2238,   199,
   -2238,  1577,   199, -2238,  2957,   199, -2238,  1583,  5826, -2238,
    1737,    87,  1737, -2238,   199,  1377,  1760,  2863,   199,   199,
   -2238,   621,  2957,   657,  2957,  1037, -2238,  1759,  1039,  5826,
   -2238,    49, -2238, -2238, -2238, -2238, -2238,  1002,   143, -2238,
     199, -2238,   457, -2238, -2238,   325, -2238,   103,  1135, -2238,
    1182, -2238,   520, -2238,   -48, -2238,   199,   199,   199, -2238,
     199,   199,   582, -2238,   199, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238,  1445,  1461,  1447,  5826, -2238,  1581,   199, -2238,
   -2238,  1762,  1764,  1767,  1769,  1770, -2238, -2238,  1771,  1773,
    1774, -2238, -2238, -2238,  1775,  1776,  1778,  1781, -2238, -2238,
   -2238,  1143,  1783,  1784,  1787,  1788,  1790, -2238, -2238, -2238,
   -2238, -2238,   199,   850, -2238, -2238,  1792, -2238,  1803, -2238,
   -2238,  1677, -2238, -2238, -2238, -2238,  5826,  2238, -2238, -2238,
   -2238,   614,   389,   389,  1567,  1568, -2238, -2238,  1578,  1582,
    1586,   633,   199, -2238, -2238, -2238, -2238,  1810, -2238, -2238,
   -2238,  1705, -2238,  1812, -2238,   159,  1807, -2238,  1811,  4131,
   -2238,  1806,  1808,  1446, -2238, -2238,  4165, -2238,  2957,  2957,
    1486, -2238,  5826,  1580,   228, -2238,   357,  2988,  2988,  2988,
     470, -2238,   501, -2238,   579, -2238,  1519,   199, -2238, -2238,
    1823,  1054, -2238,  1831, -2238,  5826, -2238, -2238, -2238,  2957,
   -2238, -2238,  2957, -2238, -2238, -2238, -2238,  5826, -2238,  1576,
    2957,  1822, -2238,  1824,  1827,  4218,  1838, -2238,   207,   199,
   -2238,  1057, -2238, -2238,  1828,  5826, -2238, -2238,  4165, -2238,
    1505, -2238,  1505,   199,   199,   199,  1077,  1093, -2238,   199,
    1834,  1829,  2957,  4255,  2878, -2238, -2238, -2238,  1519,  1285,
   -2238,  1840, -2238,  1683,  1849,  5826, -2238,   199, -2238,  1843,
    1845, -2238, -2238,  1607,  1855, -2238, -2238,  1857, -2238,  5826,
    1109, -2238,  1111, -2238, -2238,  4392, -2238, -2238,  1113, -2238,
   -2238,  5826,  1848,   199, -2238, -2238, -2238,  1856,  1859,  1664,
    1799,   199,   199,  1858,  1867, -2238,   624, -2238,  1865, -2238,
   -2238, -2238,  1866, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
     457, -2238, -2238, -2238, -2238,   325,   199, -2238, -2238,  1115,
    1872, -2238,  1873, -2238,  1875, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238,  1135, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,  1182, -2238,
   -2238, -2238, -2238, -2238,   520, -2238, -2238, -2238,   -48,  1861,
    1870,  1876,  1181,  1120, -2238,  1877,  1878,  1285, -2238,  1879,
   -2238,  1881,  1581,  1880, -2238,   199,   199, -2238,  1751,   199,
   -2238,   199,   199,   199, -2238,   199,   199,   199,  2957, -2238,
    1888,  1896, -2238,   199,   199,  2957, -2238, -2238,  1891,  2957,
    2957, -2238, -2238,  1894, -2238,  1617,   850,  2455, -2238,  1130,
   -2238,  5826, -2238, -2238, -2238,  1908, -2238, -2238, -2238, -2238,
     388,   388,   388,   388,   388,  1998, -2238,  1903,  1914,  1906,
    1998,  1807, -2238,   437,   159,   133,   133, -2238, -2238, -2238,
    1132,  1917,  1183,   506, -2238,  1921,   159, -2238,  2957, -2238,
    1913, -2238,  1446, -2238,  2409,  5826,  1918, -2238, -2238,   812,
    1911,  1919,  1139,  1920,  1922,  1923, -2238, -2238, -2238,  1927,
      13,   934, -2238,   199,  1002,  5826,    13,  5826,  1576,  2957,
    1924,  4427,  1140, -2238, -2238, -2238, -2238, -2238,  2957, -2238,
    1934, -2238, -2238, -2238, -2238, -2238,  1144,  1164,  1168, -2238,
   -2238, -2238,   689, -2238,  5826,  2957,  2957,  4461, -2238,   199,
     199, -2238, -2238,  1849, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238,  1925,    87,  1929,  3122,
   -2238,   199,   199,   199,  2863, -2238, -2238, -2238,   657, -2238,
   -2238, -2238,  2604,   199, -2238, -2238,  1858,  1942, -2238, -2238,
     199,   199,  2957, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238,   103, -2238, -2238, -2238,  2957, -2238,
    2957, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,   199,   199,
    1932,   949,  1930,  1173, -2238,  1174,   915,  1175,  1936,  1184,
    1189,  1195,  1196,  1199,  1201, -2238,  1206,  4711,  1940, -2238,
   -2238,  -127,  1213, -2238,  1215,  1217,  4742,  1166,  1954, -2238,
    5826,  5826,  1218,  1957, -2238, -2238, -2238,  1945,  4785, -2238,
   -2238, -2238,   614, -2238, -2238, -2238, -2238, -2238, -2238,  1998,
   -2238,   199, -2238, -2238,  1953,  1943, -2238,  1235,   506,   506,
     159, -2238,   159,   133,   133,   133,   133,   133,  1500,  4826,
   -2238, -2238, -2238, -2238,  2957, -2238, -2238, -2238, -2238,  2034,
   -2238,   199,  1962,  1510,   199, -2238,   199, -2238,  4857, -2238,
    2957,  2957, -2238,  4924,  1717,  2957, -2238, -2238, -2238, -2238,
    1224, -2238, -2238,  5826,  5826,  2957,  1240,  1958, -2238,  1049,
   -2238,  2957, -2238,  1951,  1955, -2238, -2238,  1963,  1972, -2238,
   -2238, -2238, -2238, -2238,  1853,  1960,  1241,  1980,  1985,  1255,
    1301,   199, -2238, -2238,  5826,    79,  1975,   368, -2238, -2238,
    1956, -2238, -2238,   304,  4955,  5000, -2238, -2238, -2238, -2238,
   -2238,  1263,  1977,   985,  2957,   199, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
    1984,  1987, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238,   199, -2238,  2957, -2238,  1653,  1266, -2238,  1273, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,  2957,  1993,
    1996, -2238, -2238, -2238,  1974,  1986, -2238,  1581, -2238,   159,
   -2238,  1500,  1988,   506,   506, -2238, -2238, -2238, -2238,  5031,
   -2238,  4165, -2238,  1278, -2238, -2238,   934,  1680, -2238,  1576,
    5826, -2238,  1750, -2238,  2001,  5088,   689, -2238,  5826, -2238,
    2549,  2003,  2004,  2006,  2007,  2009,   199,   199,  2013,  2014,
    2015,  5144, -2238, -2238, -2238,  2957,   199,   199, -2238, -2238,
    2016,   199, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
    2025, -2238, -2238, -2238, -2238, -2238,  1295, -2238, -2238, -2238,
   -2238,  2012, -2238, -2238,  2028, -2238,  5826, -2238,   199,   199,
     915, -2238,  5482, -2238, -2238,  2029,  2032, -2238,  5513, -2238,
    2035, -2238, -2238,  2022,  1303,  1500, -2238,  2957,  2034, -2238,
   -2238,  2957,   199,  2957, -2238, -2238, -2238,  5826,  1307, -2238,
   -2238,  2003,   199,   199,   199,   199, -2238, -2238,  2957,  2957,
     199,  2957,  1311, -2238, -2238,  2037, -2238,  1321,  2039,  1347,
     199,  2957, -2238, -2238, -2238, -2238,   687,  2044, -2238,  2957,
   -2238, -2238, -2238, -2238, -2238, -2238,   199,  2026, -2238, -2238,
    5544, -2238,  5826, -2238, -2238,  2038,  5575,  2549, -2238,   413,
   -2238,  2047,  1364,  2048,  1366,  2041,  1367,  5606,  5637,  2042,
   -2238,  1374,  5668, -2238,   199,  1981, -2238, -2238, -2238, -2238,
    1680, -2238,  5699,  1736, -2238,  2957,  5826,  1706,  1716, -2238,
    2056, -2238, -2238,  2957,  2134, -2238, -2238,  2064,  2065,   199,
     199, -2238,   199, -2238,  2702, -2238, -2238,  2957, -2238,   199,
   -2238,  2957, -2238,  2053,  1376,  1378, -2238,  2060,  5730,  1043,
    2061,  2062,   199,  5826,   199,  5826,  1384, -2238, -2238, -2238,
   -2238,  1386, -2238,  2063,  1393,  1395,  1400,  5761, -2238,  5826,
   -2238, -2238, -2238,  2957, -2238, -2238, -2238, -2238, -2238,  2066,
    2134, -2238,   199, -2238,  2957, -2238, -2238,  2058,  2957,  1404,
    1419,  1421,  2957,  2074, -2238, -2238, -2238,  5795,  1423, -2238,
   -2238,  5826,  2073, -2238, -2238, -2238,  2957,  2957,  2957,  2077,
   -2238, -2238, -2238,  5826, -2238, -2238,   -31,   483,  1428, -2238,
    2086,  2087, -2238, -2238, -2238,  2080,  2080,  2080, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238,   548,  2089, -2238,
    1969, -2238,  1457, -2238, -2238, -2238
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
   -2238, -2238, -2238, -2238, -2238,   -14,  1902,  1204, -2238, -2238,
    -663,   -38, -2238, -2238,  -400, -2238,   830, -2238,   -50,  -731,
   -2238, -2238, -2238,  2640,   102, -2238, -2238, -2238, -2238, -2238,
   -2238,   250,   540,   943, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238,  -165,  -901, -2238, -2238, -2238,  1040,   542,  1549,
   -2238,  -129, -1559,   260, -2238, -2238, -2238, -2238, -2238, -2238,
    1564,  -278,  -365, -2238, -2238, -2238,  1562, -2238,  -440, -2238,
   -2238, -2238, -2238,  1432, -2238, -2238,   845, -1258, -1537,  1223,
     525, -1527,  -112,    37,  1229, -2238,   257,   269, -1810, -2238,
   -1538, -1263, -1535,   -79, -2238,    63, -1555, -1299,  -805, -2238,
   -2238,   675,  1013,   441,    -1,   175, -2238,   698, -2238, -2238,
   -2238, -2238, -2238,   -15, -2238, -1452,  -615,  1151, -2238,  1133,
     767,   791,  -376, -2238, -2238,  1092, -2238, -2238, -2238, -2238,
     485,   486,  2111,  1022,  -364, -1302,   263,  -391, -1014,  1160,
    -490,  -567,  1916,  -221,  1734,  -874,  -871, -2238, -2238,  -627,
    -607,  -211, -2238,  -935, -2238,  -523,  -950, -1109, -2238, -2238,
   -2238,   246, -2238, -2238,  1482, -2238, -2238,  1948, -2238,  1949,
   -2238, -2238,   799, -2238,  -372,    11, -2238, -2238,  1959,  1964,
   -2238,   766, -2238,  -730,  -301,  1410, -2238,  1281, -2238, -2238,
     550, -2238,  1169,   571, -2238,  4496,  -409, -1080, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238,  -197, -2238,   560,  -925, -2238,
   -2238, -2238,   532, -1280,  -619,  1210,  -854,  -366,  -313,  -430,
     972,   -74, -2238, -2238, -2238,  1563, -2238, -2238,  1138, -2238,
   -2238,  1112, -2238,  1375, -1957,  1041, -2238, -2238, -2238,  1566,
   -2238,  1570, -2238,  1560, -2238,  1571,  -988, -2238, -2238, -2238,
     -94,  -236, -2238, -2238, -2238,  -403, -2238,  -223,   810,  -382,
     809, -2238,    76, -2238, -2238, -2238,  -302, -2238, -2238, -2238,
   -1843, -2238, -2238, -2238, -2238, -2238, -1432,  -517,   231, -2238,
    -144, -2238,  1433,  1221, -2238, -2238,  1225, -2238, -2238, -2238,
   -2238,  -261, -2238, -2238,  1155, -2238, -2238,  1205, -2238,   302,
    1219, -2238, -2238,  -862, -2238, -2237, -2238,  -188, -2238, -2238,
     267, -2238,  -758,  -383,  1809,  1468, -2238, -2238, -1576, -2238,
   -2238, -2238, -2238, -2238,  -134, -2238, -2238, -2238,  -274, -2238,
    -299, -2238,  -318, -2238,  -319, -1716, -1045,  -763, -2238,   -62,
    -475,  -917, -2097, -2238, -2238, -2238,  -489, -1800,   511, -2238,
    -749, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
    -485, -1469,   778, -2238,   266, -2238,  1609, -2238,  1772, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -1438,   820,
   -2238,  1512, -2238, -2238, -2238, -2238,  1895, -2238, -2238, -2238,
     331,  1869, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238,   741, -2238, -2238, -2238,   272, -2238, -2238,
   -2238, -2238,   -11, -1901, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238,   676,   477,  -526,
   -1325, -1243, -1317, -1431, -1429, -1422, -2238, -1416, -1414, -1314,
   -2238, -2238, -2238, -2238, -2238,   464, -2238, -2238, -2238, -2238,
   -2238,   507, -1413, -1405, -2238, -2238, -2238,   458, -2238, -2238,
     510, -2238,   421, -2238, -2238, -2238, -2238,   478, -2238, -2238,
   -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2238, -2238, -2238,   265, -2238,   268,   -47, -2238, -2238, -2238,
   -2238, -2238, -2238, -2238,  1082, -2238,  1425, -2238,  -843, -2238,
     253, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238, -2238,
   -2001,   -60, -2238, -2238, -2238, -2238,   752, -2238, -2238, -2238,
   -2238, -2238, -2238,    96, -2238,   751, -2238, -2238, -2238, -2238,
     745, -2238, -2238, -2238, -2238, -2238,   738, -2238,    65, -2238,
    1434
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1191
static const yytype_int16 yytable[] =
{
     410,   821,   678,   971,   829,   830,   831,   832,   817,   704,
    1391,   674,  1707,  1241,  1098,   870,  1242,  1296,  1340,  1742,
    1085,   693,  1465,   698,   422,   422,  1244,   975,  1632,   716,
     743,   430,  1337,  1337,  1230,   642,  1879,  1005,  1617,   731,
     642,   734,   800,  1612,   567,   643,   737,   738,   739,  1347,
     643,  2070,  1735,  2072,   740,  2186,  1765,   742,   792,   744,
     457,   649,   437,  1882,   653,   655,  1884,  1477,   476,   747,
    1912,  2162,   479,   402,   424,  1898,  1420,   812,   989,  2248,
    1074,  1037,  1183,  1645,  2326,   459,   773,  1182,  -565,  1809,
    2112,  1810,   793,  2324,  1902,  2327,  2328,  1044,  1811,  1917,
     789,  2329,   474,  2330,  1812,   794,  1813,  1815,  1944,   802,
     697,   805,  2331,  -541,  2332,  1816,  2333,   402,  1466,   939,
    1757,     2,     3,  1368,   824,   683,   402,  1966,  1967,  1968,
     501,  1592,  1001,  1001,  1729,  -584,   839,   570,   842,   373,
    1786,   405,   421,  1767,  1367,   848,  1241,  1231,  1904,  1386,
     898,  2573,  -563,  1537,  1283,   733,  1018,  1284,  1498,  1056,
    -593,  1558,  1339,  2450,   806,  1000,  1000,   596,   965,  1405,
     724,  1486,  -593,  1272,  1904,  1905,  1906,  1225,   667,   978,
     668,   807,  -137,   405,   725,   594,   581,   594,  2249,  -565,
    1445,  1081,   594,  1796,  -527,  1806,   589,  1819,   898,  1825,
     853,  1798,  -565,  1808,  1804,  1821,  1814,   982,  1822,   908,
     594,   984,  1057,   749,   899,  1959,   421,  1229,   991,   920,
     921,   922,   923,  2250,     8,  -527,  -141,   421,  1180,  1446,
     594,  -584,   421,   989,  1769,   -61,   963,   594,  2334,  1190,
    1058,  2335,   787,   994,   811,  1843,   421,   617,   915,   795,
     440,   642,  2518,  -339,  -541,  -339,  1212,  1758,   916,  -339,
    -563,   643,   421,  2418,  1730,   617,  1467,   974,  -593,   421,
    2065,     4,   421,  2069,   421,  1797,  1551,  1807,  1099,  1820,
    1059,  1826,  1907,  1506,  1552,   421,  1538,  2574,   421,  1539,
    1273,  1060,  1156, -1115,   497,   421,   856,   421,   858,  2266,
   -1140,  1553,   421,  1447,  1297,   859,   860,   861,  1907,  1019,
     924,   864,   865,   866,   867,  1762,   868,   869, -1062,  1061,
    1034,  1065,   421,  -137,  1029,   421,  2336,  1699,   421,  1559,
    -565,   507,  1038,  1161,   990,   669,   672,   992, -1151,  1886,
    1888,   679,   680,   684,   680, -1167,   688,   690,  1560,  2117,
     696,  1086,   700,   702,   702,   705,   594,   421,   708,  1593,
     594,   712,  1341,   708,  1645,  -594,  2115,  -141,   723,  1151,
    1597,   728,  -584,   708,  2113,   708,   -61,  -594,     5,   422,
     708,   708,   708,  2071,  2337,  2338,  2339,  1809,   708,  1810,
     741,   708,  1371,   708,   594,  2162,  1811,  1372,   508,   752,
     753,  -563,  1812,  2166,  1813,  1815,   764,   766,  1158,  -593,
    1638,   772,  1062,  1816,  1691,   989,   642,   989,   779,  1503,
     642,   783,  1497,  1437,   989,  1505,   643,   642,   736,  1148,
     643,   594,   995,   796, -1115,  1706,   803,   643,  1412,   825,
    1178, -1140,   723,   813,   815,   815,   563,   819,   796,  1554,
    1555,  1432,  1302,   568,   827,   827,   827,   827,   827, -1062,
     837,  1960,  2342,   841, -1170,   843,   779,   964, -1104,   846,
     571,   594,   849,  -594,  1177,  1001,   854,  1179,  -587, -1151,
    1796,  1608,  2211,  1672,  1454,  1150, -1167,  1152,  1798,  1300,
    -587,  1804,   594,  1806,   872,  2348,   594,  1736,   878,  1819,
     594,  1808, -1134,  1825,  1814,   871,  1294,  1821,  1365,  -591,
    1822,  2125,  1877,  1878,  1651,  1653,  1655,   421,  1301,   904,
    -382,  -591,   892,   402,  2540,  2541,   910,  2516,   575,   576,
     900,   403,  1546,   905,   906,   907,   696,   617,  1609, -1114,
    2060,   913,   914,  2126,  2189,   919,   696,   696,   696,   696,
    1295,   895,   896,   927,   928,  1355,  1356,  1357,  1358,  1359,
     594,  2477,  1797,   911,  1882,  2271,   421,  1884,  2287,  2193,
    2176,   572,  1103,  2508,  2268,  1807,   594,   573,   642, -1139,
    1001,  1820,   694,  -527,   989,  1826,  -587,  -589,   643,  1159,
    1610,  1781,  2349,  2478,  1091,  2446,   695,  1330,  2343,  -589,
   -1061,  1457,  1458,  2350, -1150, -1170,  2118,  2119, -1166, -1104,
     594,  2575,  1195,  1000,  -594,  1014,    50,  -591,   574,   712,
    1007,   764,   819,   813,   700,  1012,   594,   965,   843,  2508,
    1602,  1603,  1617,  1025,  1459,   944,   696,  1612,   597,   945,
    2576,  1895,  2111, -1134,   694,   577,   587,  1374,  1767,   985,
    2022,  1016,  2344,   986,   708,  1226,  1226,  1036,   695,   696,
    1585,  -527,  1746,  2345,  1017,  1768,   787,   960, -1169,   974,
    1896,  1375,  1680,   580,  1241,  -527,  2575,  1844,  1591,  1408,
   -1114,   988,  1053,  1829,  1830,  1831,  1376,  1835,  1836,  1630,
     965,  1377,  1378,   904,  1746,  -589,  2159,  -527,  1379,  2555,
    2577,  1098,  1098,   583,  1085,  2576,  1608,  1075,  2464,  1080,
    1093,  1767,  1082,  2562,  2375,  1347,   118,  -384, -1103,   584,
   -1139,   965,  1138,   122,  1127,  1096,  -383,  -587,  1768,  1213,
    1214,  1132,    25,   585, -1133,  1126,  1149,    29,  1547,  1769,
     586, -1061,  1770,   591,  1133, -1150,  1134,  1771,  1046, -1166,
     421,  1141,  1572,  1144,  1949,  1772,   696,  1054,  -591,    47,
      48,  1047,  1215,  1609,   642,   642,   642,   642,   642,   669,
    1055,  1174,   995,   995,   643,   643,   643,   643,   643,   592,
    1096,   588,   590,  2417,  1175,  1001,   696,   595,   989,   989,
     989,   989,   989,  1651,  1653,  1655,  -382,  1185,  1269,   787,
     593,   696,  1769,  1342,   696,  1770,  1348,  1687,   963, -1169,
    1186,  1270,  2427,    90,  1397,  1610,  1343,   985,  1000,  1349,
    1631,   986,   787,    95,  2273,  2274,   633,  1398,   638,   639,
    1233,  1413,  1380,    46,  2460,   987,  -589,  1401,  1573,  1245,
    1253,  1259,  1421,  1429,  1263,  1001,  1001,  1001,  1451,   988,
    1402,  2111,  2407,  1574,  1575,  1576,  1430,  1266,   841, -1103,
    1433,  1452,   648,  1381,  2589,  2591,    74,  1579,  1869,   121,
     985,   656,  1469,  1291,   986, -1133,  1870,  2595,  1650,  1652,
    1654,   638,   639,  1268,  1306,  1470,  1353,  1563,   987,    25,
    1564,  1565,  1566,  1472,    29,  1303,  1216,  1217,  1292,    96,
    1218,  1219,   988,  1567,  1568,   657,  1473,  1382,  1475,  1016,
    1489,  -792,  -792,  2068,   985,  1540,    47,    48,   986,  -383,
     658,  1476,  1483,  1490,  2444,   638,   639,   659,  1541,   109,
    1360,   966,   987,  1598,  1548,  1485,  1472,  2218,  1556,  2219,
    2103,  2104,  2105,  2106,  2107,   660,   988,  1549,  2220,  1550,
    2221,  1557,  -387,  1561,  2442,  1337,  1337,  1337,  1337,  1337,
    1569,  1577,   598,   661,   969,  2373,  1562,  1385,  1641,  1512,
      90,  1241,   662,  1570,  1578,  1515,  1640,  2099,   669,  1518,
      95,  1520,  1643,  1241,  1389,  1522,  2075,  1524,  -385,  1692,
     684,  1693,  1658,  2073,  2074,  1239,   599,  1996,  1046,  1638,
     969,  2084,  2085,   705,  1497,  1659,   985,  1342,  1662,   402,
     986,  1664,  1660,  2515,  1670,  2141,  1674,   638,   639,   663,
    1667,  2146,  1479,  1672,   987,  1668,   121,  1671,  1422,  1675,
    2282,   666,  1926,   728,   375,  1684,   904,  1686,   988,  1700,
     381,  1936,  2514,   675,  1448,  1753,   965,  1239,  1679,   753,
     388,  -852,  1701,   390,    25,  1690,   393,  1090,  1754,    29,
    1756,   676,  1401,   399,   766,  1239,  1241,   406,   691,  2140,
     150,   409,  1405,  2222,  2223,  1942,  2224,  2225,  1963,   699,
     944,    47,    48,   677,   945,  1239,   664,   665,  1471,   428,
    2301,   946,   947,   432,   433,   711,  1709,   714,  1969,   438,
     439,  1451,  1497,  1978,  1497,   444,   445,  2226,   447,   448,
     449,   450,   960,   451,  1970,   565,   726,  2004,   732,  1489,
    1757,  2008,   460,  2034,  1001,  1001,  1001,   464,  2054,   466,
    2005,  1501,  2006,   469,  2009,    90,  2035,   473,  2100,   475,
    2120,  2055,   995,  1761,   717,    95,   481,  -587,  2151,  2227,
     485,  2101,  1534,  2121,   488,   760,   490,  1932,  1652,  1654,
    1921,  2152,  2228,   498,   500,  2156,   985,   502,   503,   745,
     986,   750,  1534,   509,  1532,   510,  1534,   638,   639,   514,
    1842,  2215,  1534,  2232,  2302,  2157,  1542,  1543,  1544,  2158,
    -230,   121,  1534,   763,  2216,  2217,  2233,  1753,   988,  2123,
    2124,  1688,   770,  1489,  1753,  2235,   541,  1472,   543,  1472,
    2236,   782,  1571,  1929,  2241,   548,  2237,  2238,  1445,  1790,
    2239,  2251,  2240,  1472,   566,  1472,  2259,  2242,   646,  1023,
    1580,   900,  2296,   654,  2252,   150,  2253,   771,  2254,  2260,
    1721,  2023,   774,  2025,  2026,  2297,  1719,   777,  1534,  1174,
    1723,  2123,  2124,   778,  1590,  2030,  2270,  1758,  2303,   786,
     696,  2299,  2320,  1489,  1303,  2037,  1790,  2039,  1924,  2282,
    2304,  1534,  2305,  2306,  2364,  2307,  2322,   790,  2308,  1196,
    1197,  2364,  1200,   784,  2354,   791,  2378,  2365,   965,  1766,
    1780,   818,  1940,  1785,  2366,  2131,  1805,   826,  1818,  2379,
    1824,  2402,  1828,  2411,  1946,  1226,  1226,  1226,  1948,  1226,
    1226,  2120,   851,  1837,   852,  2437,  2412,  1962,   897,  1239,
    2275,  2276,  2277,  2218,  2429,  2219,  1767,   787,  2438,  1489,
     855,  1447,  2453,   944,  2220,  1656,  2221,   945,   607,  1597,
     857,  1241,  2456,  1768,   946,   947,  1933,  1934,  1935, -1190,
     948,   949,   950,   951,   952,  2458,   953,   954,   955,   956,
     957,   958,   989,   959,  2381,   960,   961,  1241,  2459,  2309,
    2361,  2310,  1451,  1767,  1489,  1753,  1588,  1589,  1682,  1682,
     876,  1682,  2489,   877,  2232,  2481,  2232,  2483,  2485,   708,
    1768,  1075,  2530,  1075,  2532,  2490,  1694,  2521,   880,  2522,
    1791,  1451,   887,  1489,   888,  2531,  1192,  2533,  1174,   889,
    1198,   621,  1239,   890,  2535,  1204,  2536,  1769,  1208,  1210,
    1770,  2537,   642,   995,  1792,  2552,   720,  2364,  1717,  2364,
     421,  2558,   643,  1733,  1793,   891,  2578,  1144,   909,  1794,
    2553,   894,  2554,  1731,  2559,   966,  1048,  1791,  1049,  2579,
     912,   607,  1759,   696,  1760,   608,   609,   610,   611,  2222,
    2223,   969,  2224,  2225,  1769,  1634,   967,  1770,   612,   936,
    1763,  1792,  1771,   972,  1298,   613,  1299,   614,  2594,  1832,
    1772,  1793,  1226,  1226,  1226,   968,  1794,   799,   638,   639,
    1233,  1344,   983,  2226,  1860,  1861,   607,  2052,  2053,   612,
     608,   609,   610,   611,   993,   615,  1245,   807,   614,  2539,
    1013,  1253,  1022,   612,  2256,  2257,  2123,  2124,  1028,  1259,
     613,  1024,   614,  1887,  1889,  1361,  1362,  1263,   605,   606,
     847,  2584,  2585,   616,  1026,  2227,   615,  1127,  1035,  1875,
    2494,  2495,  1206,  1207,  1868,  1041,  1045,   405,  2228,  1071,
     615,  1069,  1066,   963,  2028,  1087,  1088,  1089,  1094,  2031,
    1102,  1104,  1157,  1168,   616,  2142,  1131,  1181,  2404,  1171,
     421,  1172,  2041,  1187,  1897,  1195,  1201,  1224,   616,  1184,
    2144,  1216,  1239,  1262,  1271,   976,  2131,  1908,  1307,   977,
     617,   594,  1310,  1331,  2043,   799,   638,   639,   945,  1332,
    2045,   640,  -342,   986,  2047,  1363,  1694,   612,  1369,  1694,
    1694,  1694,   775,  1370,  1390,   421,   614,  1414,  1415,  1937,
    1392,   617,  1400,  1416,   787,  1226,  1417,  1418,  1226,  1419,
    2188,  1428,  1431,  1434,  1440,   617,   787,   558,   559,   560,
     561,   562,   618,  1098,   615,  2445,  1497,   822,   651,  1436,
    1449,   708,   978,  -935,  1450,  1455,  1461,   834,   637,   638,
     639,  1274,  1456,  1460,  1468,  1474,  1484,  1487,  1488,  1492,
     612,  1971,   616,   618,   944,  1491,  1494,  1493,   945,   614,
    1502,  1504,   915,  1526,  1511,   946,   947,   618,  1534,  1082,
    1527,   948,   949,  1528,   951,  1514,  1529,   953,   954,   955,
     956,   957,  1530,  1531,  1275,  2492,   960,   615,  1582,   640,
     881,   882,   883,   884,   886,  2323,  1535,  1536,  1587,   787,
    1625,  1633,  1628,  2015,  2016,   787,  1634,  1639,   373,   617,
    2513,  1098,  1644,  1647,  2264,   616,  1277,  1648,  1649,  1657,
    1663,   403,   978,  1513,  2267,  1728,   978,  1680,  2033,  1516,
    1689,  1702,  1697,  1519,  1703,  1521,  1519,  1519,  1226,  1523,
    1278,  1525,  1279,  1704,  1705,  1446,  1716,  1737,  1721,   938,
     940,   941,   640,   942,  1755,  1839,  1840,  1845,  1841,  1846,
     943,   618,  1847,  2513,  1848,  1849,  1851,  2184,  1852,  1853,
    1855,  1856,   617,  1857,   822,  1280,  1858,  2066,  1862,  1863,
    1281,  2195,  1864,  1865,  1282,  1866,  2198,  1872,  1283,  2199,
    1874,  1284,  2081,  1890,  1891,  1900,  2087,  2396,  2397,  2206,
    1903,  2207,  1916,  2208,  1892,  2209,  1918,  1921,  1893,  1922,
    1350,  1941,  1894,  1285,  1351,   607,  1021,  1226,  1943,   608,
     609,   610,   611,  1954,   618,  1955,  1958,  1286,  1956,  1972,
    1973,  1031,   612,  1287,  -494,  1979,  1980,  1981,  1998,   613,
    1999,   614,  2001,  2002,  2003,  2114,  1908,  1908,  1908,  2010,
    1757,  2013,  1758,   635,  2014,  2021,  2020,   636,  1908,  1052,
    2024,  2027,  2049,   637,   638,   639,  1068,  2036,  2038,   615,
    2040,  2050,  1072,  1073,  2058,   612,  2059,  2051,  2056,  2057,
     617,  2061,   944,  2079,   614,   684,   945,  2441,  2443,  2154,
    2380,  2080,  2089,   946,   947,  2094,  2102,   616,  -381,  2109,
     949,  2110,   951,  1226,  2122,   953,   954,   955,   956,   957,
    2127,  2169,   615,  2130,   960,  1293,   798,  2134,  1130,  2133,
    2135,  2136,  2139,  2137,  2138,  2149,   799,   638,   639,  2155,
    2191,  2171,  1669,  2213,   421,  2173,  2214,  2234,   612,  1731,
     616,  2244,  2258,  1937,  2261,  2177,  2262,   614,  -224,  2269,
    2286,  2294,  2313,  2300,   617,  2187,  2314,  2197,  2315,  1594,
    2316,  2319,  2192,  1763,   607,  2317,  2203,  -843,   608,   609,
     610,   611,  2321,  2347,   787,   615,  2341,   640,  2355,  2358,
    2369,   612,  2359,  2370,  1832,  2382,  2363,  2372,   613,  2376,
     614,  1188,  1189,  1601,  1602,  1603,  2383,   617,  2390,  2392,
     787,  2393,  2394,   616,  2395,  1604,   618,  1727,  2398,  2399,
    2400,  2406,  2410,  2414,  1605,  1373,  2415,  2421,   615,  2280,
    2423,  1463,  2426,  2428,   607,  2454,  2457,  2472,   608,   609,
     610,   611,  2465,  2474,  2480,  2482,  2484,  1606,  2500,  2493,
     640,   612,  1607,  2488,  2502,  2497,   616,  2501,   613,   618,
     614,  2509,  2510,  1897,  2520,  2523,  2525,  2526,  2534,  2550,
     617,  2560,  1908,  2542,  1908,  1908,  1908,  1908,  1908,  1908,
    1608,  2556,  2565,  2580,  2581,  2582,  2593,  2575,   615,  1581,
    1267,  2174,  2292,   421,  1311,  1312,  1313,  1314,  1315,  1316,
    1317,  1318,  1319,  1320,  1321,  1322,  1323,  1324,  1325,  1326,
    1327,  1328,  1329,   617,   578,  2095,   616,  1873,  1507,   822,
    1444,   917,   618,  1876,  1345,  2371,  2096,  2416,  1226,  1746,
     978,  1352,  1042,  1354,   607,  1600,  1901,  1609,   608,   609,
     610,   611,   815,  2016,   926,   929,  1304,  2374,  1364,  2272,
    2116,   612,  1305,   421,  2108,  2265,  1744,  1482,   613,  1997,
     614,  2312,  2172,  1726,  2504,   618,  2325,  1393,  1676,  1410,
    1439,  1661,  2360,   617,   886,  1964,   436,  2132,  1965,  1226,
    1395,  2592,  2145,  1043,   600,   601,  1666,   788,   615,  1610,
    1696,  2081,  1147,  1928,  1399,   602,  1052,  1048,  1366,  1425,
     603,  1927,   607,  1176,  1453,  1500,   608,   609,   610,   611,
    1006,  1010,  2403,  1678,  1681,  1008,   616,  2291,  2545,   612,
    1009,  1908,  2175,  1011,  2431,   618,   613,  1384,   614,  1383,
    1155,  2527,  1438,  1880,  2499,  2143,  1403,  1394,   607,  2476,
    2170,   761,   608,   609,   610,   611,  1139,  2440,  2546,  2564,
    2586,  2588,  2409,   421,  2011,   612,   615,  1743,   827,   827,
    2182,  1708,   613,  1095,   614,  1015,   845,   687,  2177,  2147,
    2183,  1838,  2042,   617,   735,  2385,  2048,  2029,  2046,    13,
      14,  1464,    15,    16,   616,  2032,  2044,    20,  1586,  2202,
    2413,  2200,   615,  1232,  2420,    23,  1533,  2210,  1850,  1478,
      27,  2357,  1854,    30,  1859,  1867,  2367,     0,     0,     0,
       0,    37,     0,    38,     0,    40,     0,     0,     0,     0,
     616,   421,   886,  1499,   834,   618,     0,     0,     0,     0,
       0,     0,     0,     0,  2433,     0,     0,     0,    59,     0,
     748,   617,     0,     0,  1096,  1096,  2455,     0,     0,    70,
       0,     0,     0,     0,     0,  2463,     0,   421,     0,     0,
       0,  2467,     0,  2468,     0,     0,     0,     0,     0,   748,
       0,    13,    14,    85,    15,    16,   748,   617,  2469,    20,
       0,     0,     0,     0,   808,     0,    93,    23,     0,  1545,
       0,     0,    27,   618,     0,    30,     0,  2066,  2066,     0,
       0,     0,     0,    37,   102,    38,  1463,    40,     0,   607,
     104,     0,     0,   608,   609,   610,   611,     0,   108,     0,
     110,     0,   112,     0,   114,     0,   612,     0,     0,   618,
      59,   119,     0,   613,     0,   614,     0,     0,     0,     0,
       0,    70,     0,     0,     0,  1596,     0,     0,   130,   131,
    2097,     0,     0,     0,  2469,   607,  2528,  1629,     0,   608,
     609,   610,   611,   615,     0,    85,   143,     0,     0,     0,
       0,     0,   612,   604,     0,     0,     0,     0,    93,   613,
       0,   614,     0,     0,     0,  1636,     0,   155,     0,     0,
     156,   616,     0,     0,     0,   925,   102,  1642,     0,     0,
    2566,     0,   104,     0,     0,  2567,     0,     0,     0,   615,
     108,     0,   110,     0,   112,     0,   114,     0,  2567,     0,
    2587,  2590,     0,   119,     0,     0,     0,     0,   421,     0,
       0,     0,     0,  2590,     0,     0,     0,   616,     0,     0,
     130,   131,     0,     0,  2386,     0,  1665,  1031,   617,   607,
       0,     0,     0,   608,   609,   610,   611,     0,   143,   748,
       0,  1677,     0,     0,     0,   978,   612,     0,  1685,     0,
       0,     0,     0,   613,   421,   614,     0,     0,   886,   155,
       0,     0,   156,     0,  1695,     0,  1698,     0,     0,     0,
       0,     0,     0,     0,   617,     0,     0,     0,     0,   886,
     618,   833,     0,   615,   607,     0,     0,     0,   608,   609,
     610,   611,     0,  1713,     0,     0,     0,     0,     0,     0,
       0,   612,     0,     0,  1725,     0,     0,     0,   613,   944,
     614,   616,     0,   945,     0,     0,     0,  1739,     0,     0,
     946,   947,  1745,     0,  1751,     0,   618,   949,     0, -1191,
       0,  1078, -1191, -1191, -1191, -1191, -1191,     0,   615,     0,
       0,   960,     0,     0,     0,     0,     0,     0,   421,  1097,
       0,     0,  2185,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,  1129,     0,     0,   616,     0,   617,     0,
       0,  1135,  1136,     0,     0,     0,  1142,     0,     0,   833,
       0,     0,   607,     0,     0,     0,   608,   609,   610,   611,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   612,
       0,     0,     0,   421,  1097,     0,   613,     0,   614,     0,
       0,     0,  1169,     0,     0,     0,     0,     0,     0,     0,
     618,     0,     0,   617,     0,     0,     0,  1881,     0,     0,
    1030,     0,     0,   607,     0,     0,   615,   608,   609,   610,
     611,  1191,     0,  1194,     0,     0,     0,     0,     0,     0,
     612,     0,     0,     0,     0,     0,     0,   613,     0,   614,
       0,     0,   748,   748,   616,     0,     0,     0,  1925,   822,
    1345,     0,     0,  1240,  1142,   618,  1463,     0,     0,   607,
       0,     0,     0,   608,   609,   610,   611,   615,     0,     0,
       0,     0,     0,     0,     0,     0,   612,     0,     0,  1945,
       0,   421,  1947,   613,     0,   614,     0,     0,     0,  1712,
    1951,     0,   607,     0,     0,   616,   608,   609,   610,   611,
       0,   617,     0,     0,     0,     0,     0,     0,     0,   612,
       0,     0,     0,   615,     0,     0,   613,     0,   614,     0,
    1738,     0,  1974,   607,  1977,     0,     0,   608,   609,   610,
     611,     0,   421,  1336,  1336,  1976,     0,     0,   607,  1274,
     612,   616,   608,   609,   610,   611,   615,   613,     0,   614,
       0,     0,   617,   618,     0,   612,     0,     0,     0,   748,
     748,     0,   613,     0,   614,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   616,     0,     0,   615,   421,     0,
       0,     0,  1275,     0,     0,     0,  1240,     0,  1276,     0,
       0,     0,   615,     0,     0,     0,     0,     0,   617,     0,
       0,     0,     0,     0,   618,   616,     0,     0,     0,  1404,
       0,   421,     0,     0,  1277,     0,     0,   607,     0,     0,
     616,   608,   609,   610,   611,     0,     0,     0,     0,     0,
       0,   617,     0,     0,   612,     0,     0,     0,  1278,     0,
    1279,   613,   421,   614,     0,     0,     0,   748,   607,     0,
     618,     0,   608,   998,   610,   611,     0,   421,     0,     0,
       0,     0,   617,     0,     0,   612,     0,     0,  2077,     0,
       0,   615,   613,  1280,   614,  2086,     0,   617,  1281,  2090,
    2091,     0,  1282,   618,     0,     0,  1283,  2098,     0,  1284,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   616,
       0,     0,   615,     0,     0,     0,     0,     0,     0,     0,
       0,  1285,     0,     0,   618,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,  1286,     0,     0,  2129,   618,
     616,  1287,     0,     0,  1636,     0,   421,  1097,     0,     0,
       0,     0,     0,     0,     0,     0,   944,     0,  1714,     0,
     945,   607,     0,     0,     0,     0,   617,   946,   947,  2148,
       0,     0,  1715,   948,   949,   950,   951,   421,  2153,   953,
     954,   955,   956,   957,   958,     0,   959,  1105,   960,   961,
       0,  1106,   607,     0,     0,  2163,  2164,   617,  1107,  1108,
       0,     0,     0,     0,  1109,  1110,  1111,  1112,     0,     0,
    1113,  1114,  1115,  1116,  1117,  1118,  1119,  1120,   618,  1121,
    1122,     0,     0,     0,  1739,     0,     0,     0,     0,     0,
       0,     0,   834,     0,     0,     0,     0,     0,     0,     0,
       0,     0,  2194,     0,     0,     0,     0,     0,     0,   618,
       0,     0,     0,     0,     0,     0,     0,     0,  2204,   944,
    2205,  1173,     0,   945,   607,     0,     0,     0,     0,     0,
     946,   947,     0,     0,     0,  1599,   948,   949,   950,   951,
       0,     0,   953,   954,   955,   956,   957,   958,     0,   959,
     944,   960,   961,  1236,   945,   607,     0,     0,     0,     0,
       0,   946,   947,     0,     0,     0,     0,   948,   949,   950,
     951,     0,     0,   953,   954,   955,   956,   957,   958,     0,
     959,   962,   960,   961,     0,   944,     0,     0,  1237,   945,
     607,     0,     0,     0,     0,     0,   946,   947,     0,   748,
       0,     0,   948,   949,   950,   951,     0,     0,   953,   954,
     955,   956,   957,   958,  2279,   959,     0,   960,   961,  2281,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   944,
    2290,  1951,  1238,   945,   607,  2295,     0,     0,     0,     0,
     946,   947,     0,     0,     0,  2298,   948,   949,   950,   951,
       0,  2311,   953,   954,   955,   956,   957,   958,     0,   959,
       0,   960,   961,     0,     0,  1078,     0,  1078,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,  1097,  1097,     0,   944,
       0,     0,     0,   945,  2356,     0,     0,  1710,     0,     0,
     946,   947,     0,   748,     0,  1720,   948,   949,  1722,   951,
       0,     0,   953,   954,   955,   956,   957,     0,     0,   959,
     808,   960,   961,  2362,  1097,  1097,     0,     0,     0,     0,
       0,  1123,     0,     0,     0,   962,     0,     0,  2368,     0,
       0,     0,     0,     0,  1596,     0,   748,     0,     0,     0,
       0,     0,     0,  1787,     0,     0,     0,     0,     0,     0,
       0,     0,   748,   748,   748,     0,   748,   748,     0,     0,
     748,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    2387,   944,   962,     0,  1240,   945,   607,     0,     0,     0,
       0,     0,   946,   947,   962,   886,     0,  1309,   948,   949,
     950,   951,     0,     0,   953,   954,   955,   956,   957,   958,
       0,   959,     0,   960,   961,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   962,   962,   962,   962,     0,   962,     0,  1899,     0,
       0,     0,     0,     0,     0,     0,     0,  2430,  2281,     0,
       0,  2432,     0,  2436,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,  2447,  2448,
     748,  2452,     0,     0,     0,     0,     0,     0,     0,     0,
       0,  2462,     0,     0,     0,     0,     0,     0,   962,  2466,
     962,   962,   962,   962,   944,     0,     0,     0,   945,   607,
       0,     0,     0,     0,     0,   946,   947,  2387,     0,     0,
    1388,   948,   949,   950,   951,     0,     0,   953,   954,   955,
     956,   957,   958,     0,   959,     0,   960,   961,     0,   748,
     748,   748,     0,     0,     0,  2498,     0,     0,     0,     0,
       0,     0,     0,  2503,  2505,     0,     0,     0,     0,   962,
       0,     0,     0,     0,   834,     0,     0,  2517,     0,     0,
       0,  2519,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   962,     0,     0,     0,     0,     0,     0,     0,  1097,
       0,   962,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   886,     0,     0,     0,     0,     0,     0,
    2505,     0,   962,     0,  2547,     0,     0,     0,  2551,     0,
       0,     0,  1751,     0,     0,     0,     0,     0,   962,     0,
       0,     0,   962,   962,     0,     0,  1751,  2563,  2547,   944,
       0,  1396,     0,   945,   607,     0,     0,     0,     0,     0,
     946,   947,     0,     0,     0,     0,   948,   949,   950,   951,
       0,     0,   953,   954,   955,   956,   957,   958,     0,   959,
       0,   960,   961,     0,     0,     0,     0,     0,     0,     0,
       0,  1240,   748,     0,     0,   748,     0,  1097,  1097,  1097,
     962,  1142,  1142,  1240,     0,     0,     0,     0,     0,  1142,
    1142,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,  1336,  1336,  1336,  1336,
    1336,     0,     0,   944,     0,  1411,     0,   945,   607,     0,
       0,     0,     0,     0,   946,   947,     0,     0,   962,   962,
     948,   949,   950,   951,     0,     0,   953,   954,   955,   956,
     957,   958,     0,   959,     0,   960,   961,   944,     0,     0,
       0,   945,   607,     0,     0,     0,  1240,     0,   946,   947,
       0,     0,  1404,  1423,   948,   949,   950,   951,     0,     0,
     953,   954,   955,   956,   957,   958,     0,   959,     0,   960,
     961,   944,     0,     0,     0,   945,   607,     0,     0,     0,
       0,     0,   946,   947,     0,   748,  2167,  1426,   948,   949,
     950,   951,     0,     0,   953,   954,   955,   956,   957,   958,
       0,   959,     0,   960,   961,   944,     0,     0,   808,   945,
     607,     0,     0,     0,     0,     0,   946,   947,     0,  1097,
       0,  1427,   948,   949,   950,   951,     0,     0,   953,   954,
     955,   956,   957,   958,     0,   959,     0,   960,   961,     0,
    2201,   962,   962,   962,   962,   962,   962,   962,   962,   962,
     962,   962,   962,   962,   962,   962,   962,   962,   962,   962,
       0,     0,     0,     0,   748,  2212,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   962,     0,     0,     0,     0,
       0,   944,   962,     0,   962,   945,   607,     0,     0,     0,
       0,     0,   946,   947,   962,     0,     0,  1462,   948,   949,
     950,   951,     0,     0,   953,   954,   955,   956,   957,   958,
       0,   959,     0,   960,   961,     0,   944,  1899,     0,     0,
     945,   607,     0,     0,     0,   962,     0,   946,   947,     0,
       0,     0,  1509,   948,   949,   950,   951,     0,     0,   953,
     954,   955,   956,   957,   958,     0,   959,  2285,   960,   961,
     748,   944,  2288,     0,     0,   945,   607,     0,     0,     0,
       0,     0,   946,   947,     0,     0,     0,  1510,   948,   949,
     950,   951,     0,     0,   953,   954,   955,   956,   957,   958,
       0,   959,     0,   960,   961,     0,   944,     0,     0,     0,
     945,   607,     0,     0,   962,     0,     0,   946,   947,     0,
       0,     0,  1646,   948,   949,   950,   951,     0,   962,   953,
     954,   955,   956,   957,   958,     0,   959,     0,   960,   961,
       0,  1240,     0,     0,     0,     0,   944,     0,     0,   962,
     945,   607,     0,     0,     0,     0,     0,   946,   947,     0,
       0,     0,  1919,   948,   949,   950,   951,  1240,     0,   953,
     954,   955,   956,   957,   958,     0,   959,     0,   960,   961,
     944,     0,  1714,     0,   945,   607,     0,     0,     0,     0,
       0,   946,   947,     0,     0,   962,     0,   948,   949,   950,
     951,     0,     0,   953,   954,   955,   956,   957,   958,     0,
     959,     0,   960,   961,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   944,     0,     0,     0,   945,   607,     0,
       0,     0,     0,  1097,   946,   947,   962,  1097,     0,  1957,
     948,   949,   950,   951,     0,     0,   953,   954,   955,   956,
     957,   958,     0,   959,     0,   960,   961,     0,     0,     0,
     944,     0,  1975,     0,   945,   607,     0,     0,     0,   962,
       0,   946,   947,     0,   748,   748,   962,   948,   949,   950,
     951,     0,   962,   953,   954,   955,   956,   957,   958,     0,
     959,     0,   960,   961,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   962,     0,     0,  1097,  1097,
    1097,  1097,     0,     0,     0,     0,  2167,   962,     0,     0,
       0,     0,     0,     0,     0,   962,   748,     0,     0,     0,
       0,     0,     0,     0,     0,   962,     0,     0,   962,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   962,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   962,     0,     0,     0,     0,
    1097,     0,     0,     0,     0,     0,     0,     0,     0,   962,
       0,     0,     0,     0,     0,   962,     0,     0,     0,     0,
       0,   962,     0,     0,     0,  1097,  1097,   944,  1097,     0,
       0,   945,   607,     0,     0,  2167,     0,     0,   946,   947,
       0,     0,     0,  2007,   948,   949,   950,   951,     0,     0,
     953,   954,   955,   956,   957,   958,     0,   959,     0,   960,
     961,     0,   944,     0,  2150,     0,   945,   607,     0,     0,
       0,     0,     0,   946,   947,     0,     0,     0,  1097,   948,
     949,   950,   951,     0,     0,   953,   954,   955,   956,   957,
     958,     0,   959,     0,   960,   961,   944,     0,  2165,     0,
     945,   607,     0,     0,     0,     0,     0,   946,   947,     0,
       0,     0,     0,   948,   949,   950,   951,     0,     0,   953,
     954,   955,   956,   957,   958,     0,   959,     0,   960,   961,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     377,   378,   379,   380,     0,   382,     0,   383,     0,   385,
     386,   962,   387,     0,     0,     0,   389,     0,   391,   392,
       0,     0,   394,   395,   396,     0,   398,     0,   400,   401,
       0,     0,   408,     0,     0,     0,     0,   411,   412,   413,
     414,   415,   416,   417,   418,     0,   419,   420,     0,     0,
     425,   426,   427,     0,   429,   962,   431,     0,     0,   434,
     435,     0,     0,     0,     0,     0,     0,   442,   443,     0,
       0,   446,     0,     0,     0,   962,     0,   962,   452,     0,
     454,   962,   455,   456,     0,     0,     0,   461,   462,   463,
       0,     0,   465,     0,     0,   467,   468,     0,   470,   471,
     472,     0,     0,     0,   962,   477,   478,   962,     0,   480,
       0,   482,   483,   484,     0,   486,   487,     0,     0,   489,
       0,   491,   492,   493,   494,   495,   496,     0,     0,  1123,
       0,     0,     0,   504,   505,   506,     0,     0,     0,     0,
     511,   512,   513,     0,   515,   516,   517,   518,   519,   520,
     521,   522,   523,   524,   525,   526,   527,   528,   529,   530,
     531,   532,   533,   534,   535,   536,   537,   538,   539,   540,
       0,   542,     0,   544,     0,   545,     0,   546,   547,     0,
     549,   550,   551,   552,   553,   554,   555,   556,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   944,   962,     0,  2243,
     945,   607,     0,     0,     0,     0,   962,   946,   947,     0,
     962,   962,     0,   948,   949,   950,   951,     0,   962,   953,
     954,   955,   956,   957,   958,     0,   959,   944,   960,   961,
     634,   945,   607,   647,     0,   650,     0,     0,   946,   947,
       0,     0,     0,  2255,   948,   949,   950,   951,     0,   962,
     953,   954,   955,   956,   957,   958,     0,   959,     0,   960,
     961,     0,     0,     0,     0,     0,     0,     0,   962,     0,
     944,     0,     0,   962,   945,   607,     0,     0,     0,     0,
       0,   946,   947,   962,   962,     0,  2263,   948,   949,   950,
     951,     0,     0,   953,   954,   955,   956,   957,   958,     0,
     959,     0,   960,   961,     0,     0,     0,     0,     0,     0,
       0,   944,     0,     0,   962,   945,   607,     0,     0,     0,
       0,     0,   946,   947,   962,   962,     0,  2278,   948,   949,
     950,   951,     0,     0,   953,   954,   955,   956,   957,   958,
       0,   959,   944,   960,   961,  2289,   945,   607,     0,     0,
       0,     0,     0,   946,   947,     0,     0,     0,     0,   948,
     949,   950,   951,     0,     0,   953,   954,   955,   956,   957,
     958,     0,   959,     0,   960,   961,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   962,
       0,   962,     0,     0,     0,     0,     0,     0,     0,   944,
     962,     0,     0,   945,   607,   962,     0,     0,   962,     0,
     946,   947,     0,     0,     0,  2293,   948,   949,   950,   951,
       0,   962,   953,   954,   955,   956,   957,   958,     0,   959,
     944,   960,   961,     0,   945,   607,     0,     0,     0,     0,
       0,   946,   947,     0,     0,     0,  2352,   948,   949,   950,
     951,     0,     0,   953,   954,   955,   956,   957,   958,     0,
     959,     0,   960,   961,     0,     0,   962,     0,     0,     0,
       0,     0,   962,     0,     0,   944,     0,     0,   962,   945,
     607,     0,     0,     0,     0,     0,   946,   947,     0,     0,
       0,  2353,   948,   949,   950,   951,     0,   962,   953,   954,
     955,   956,   957,   958,     0,   959,   944,   960,   961,  2377,
     945,   607,     0,     0,     0,     0,     0,   946,   947,     0,
       0,     0,     0,   948,   949,   950,   951,     0,     0,   953,
     954,   955,   956,   957,   958,     0,   959,     0,   960,   961,
     962,     0,   962,     0,     0,     0,   962,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   962,   962,     0,
       0,     0,   962,   944,   934,   935,     0,   945,   607,     0,
       0,     0,   962,     0,   946,   947,   962,     0,     0,  2384,
     948,   949,   950,   951,     0,     0,   953,   954,   955,   956,
     957,   958,     0,   959,     0,   960,   961,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   962,     0,
       0,     0,     0,   962,     0,   962,     0,     0,     0,   944,
       0,     0,  2401,   945,   607,     9,     0,   962,     0,   962,
     946,   947,    10,     0,     0,     0,   948,   949,   950,   951,
       0,     0,   953,   954,   955,   956,   957,   958,     0,   959,
       0,   960,   961,     0,     0,     0,     0,   962,     0,     0,
       0,   962,    11,    12,    13,    14,     0,    15,    16,    17,
      18,    19,    20,   962,     0,    21,    22,     0,     0,     0,
      23,    24,    25,     0,    26,    27,    28,    29,    30,    31,
       0,    32,    33,    34,    35,    36,    37,     0,    38,    39,
      40,    41,    42,    43,     0,     0,    44,    45,    46,    47,
      48,     0,     0,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
      66,    67,    68,    69,    70,     0,    71,     0,    72,    73,
       0,    74,    75,    76,     0,     0,    77,     0,     0,    78,
      79,     0,    80,    81,    82,    83,     0,    84,    85,    86,
      87,    88,    89,    90,    91,    92,     0,     0,     0,     0,
       0,    93,    94,    95,    96,     0,     0,     0,     0,    97,
       0,     0,    98,    99,     0,     0,   100,   101,     0,   102,
       0,     0,     0,   103,     0,   104,     0,   105,     0,     0,
       0,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,     0,   116,   117,   118,     0,   119,     0,   120,   121,
       0,   122,     0,   123,   124,   125,   126,     0,     0,   127,
     128,   129,     0,   130,   131,   132,     0,   133,   134,   135,
       0,   136,     0,   137,   138,   139,   140,   141,     0,   142,
       0,   143,   144,     0,     0,   145,   146,   147,     0,     0,
     148,   149,     0,   150,   151,     0,   152,   153,     0,     0,
       0,   154,   155,     0,     0,   156,     0,     0,   157,     0,
       0,     0,   158,   159,     0,     0,   160,   161,   162,     0,
     163,   164,   165,   166,   167,   168,   169,   170,   171,   172,
     173,     0,   174,     0,     0,   175,     0,     0,     0,   176,
     177,   178,   179,   180,     0,   181,   182,     0,     0,   183,
     184,   185,   186,     0,     0,     0,     0,   187,   188,   189,
     190,   191,   192,     0,     0,     0,     0,     0,     0,     0,
     193,     0,     0,   194,   195,   196,   197,   198,     0,     0,
       0,   199,   200,   201,   202,   203,   204,   944,   205,   206,
    2419,   945,   607,   207,     0,     0,     0,     0,   946,   947,
       0,     0,     0,     0,   948,   949,   950,   951,     0,     0,
     953,   954,   955,   956,   957,   958,     0,   959,   944,   960,
     961,     0,   945,   607,     0,     0,     0,     0,     0,   946,
     947,     0,     0,     0,  2425,   948,   949,   950,   951,     0,
       0,   953,   954,   955,   956,   957,   958,     0,   959,   944,
     960,   961,  2473,   945,   607,     0,     0,     0,     0,     0,
     946,   947,     0,     0,     0,     0,   948,   949,   950,   951,
       0,     0,   953,   954,   955,   956,   957,   958,     0,   959,
     944,   960,   961,     0,   945,   607,     0,     0,     0,     0,
       0,   946,   947,     0,     0,     0,  2475,   948,   949,   950,
     951,     0,     0,   953,   954,   955,   956,   957,   958,     0,
     959,   944,   960,   961,     0,   945,   607,     0,     0,     0,
       0,     0,   946,   947,     0,     0,     0,  2486,   948,   949,
     950,   951,     0,     0,   953,   954,   955,   956,   957,   958,
       0,   959,   944,   960,   961,  2487,   945,   607,     0,     0,
       0,     0,     0,   946,   947,     0,     0,     0,     0,   948,
     949,   950,   951,     0,     0,   953,   954,   955,   956,   957,
     958,     0,   959,   944,   960,   961,  2491,   945,   607,     0,
       0,     0,     0,     0,   946,   947,     0,     0,     0,     0,
     948,   949,   950,   951,     0,     0,   953,   954,   955,   956,
     957,   958,     0,   959,   944,   960,   961,     0,   945,   607,
       0,     0,     0,     0,     0,   946,   947,     0,     0,     0,
    2496,   948,   949,   950,   951,     0,     0,   953,   954,   955,
     956,   957,   958,     0,   959,   944,   960,   961,     0,   945,
     607,     0,     0,     0,     0,     0,   946,   947,     0,     0,
       0,  2524,   948,   949,   950,   951,     0,     0,   953,   954,
     955,   956,   957,   958,     0,   959,   944,   960,   961,  2538,
     945,   607,     0,     0,     0,     0,     0,   946,   947,     0,
       0,     0,     0,   948,   949,   950,   951,     0,     0,   953,
     954,   955,   956,   957,   958,     0,   959,     0,   960,   961,
     944,     0,  2557,     0,   945,   607,     0,     0,     0,     0,
       0,   946,   947,     0,     0,     0,     0,   948,   949,   950,
     951,     0,     0,   953,   954,   955,   956,   957,   958,     0,
     959,   944,   960,   961,     0,   945,   607,     0,     0,     0,
       0,     0,   946,   947,     0,     0,     0,     0,   948,   949,
     950,   951,     0,     0,   953,   954,   955,   956,   957,   958,
       0,   959,   944,   960,   961,     0,   945,     0,     0,     0,
       0,     0,     0,   946,   947,     0,     0,     0,     0,   948,
     949,   950,   951,     0,     0,   953,   954,   955,   956,   957,
     958,     0,   959,     0,   960,   961
};

static const yytype_int16 yycheck[] =
{
      50,   486,   378,   630,   493,   494,   495,   496,   483,   392,
    1024,   375,  1450,   887,   763,   541,   887,   918,   968,  1488,
     750,   387,  1131,   388,    62,    63,   888,   634,  1330,   401,
     430,    69,   967,   968,   877,   256,  1595,   656,  1301,   411,
     261,   413,   472,  1301,   209,   256,   418,   419,   420,   974,
     261,  1851,  1484,  1853,   426,  2012,  1508,   429,   467,   431,
      98,   258,    76,  1601,   261,   262,  1601,  1147,   118,   433,
    1625,  1972,   122,     7,    63,  1612,  1064,   480,   645,  2080,
     743,   696,   845,  1363,     5,    99,   450,   845,    15,  1520,
    1900,  1520,   468,  2190,  1621,    16,    17,   712,  1520,  1626,
     464,    22,   116,    24,  1520,   469,  1520,  1520,  1667,   473,
     388,   475,    33,   116,    35,  1520,    37,     7,   175,   609,
      71,     0,     1,     8,   488,     8,     7,  1703,  1704,  1705,
     144,    15,   655,   656,    47,    20,   500,   211,   502,    15,
      37,    15,   129,   191,   998,   509,  1020,   878,    15,  1020,
       5,   182,     8,   182,   191,    15,   673,   194,  1172,     5,
       8,    41,    64,  2400,   477,   655,   656,   241,     3,  1043,
     406,  1159,    20,   904,    15,    16,    17,    21,     5,   166,
       7,    15,   116,    15,   407,     8,   224,     8,   315,   116,
     169,    26,     8,  1518,    15,  1520,     8,  1522,     5,  1524,
     513,  1518,   129,  1520,  1518,  1522,  1520,   637,  1522,   574,
       8,   641,    58,   434,    21,     8,   129,    21,   648,   584,
     585,   586,   587,   350,   116,    37,   116,   129,   843,   208,
       8,   116,   129,   800,   282,   116,    15,     8,   159,   854,
      86,   162,   463,    15,   480,  1547,   129,   149,     5,   470,
      64,   472,  2489,   129,   257,   129,   871,   208,    15,   129,
     116,   472,   129,  2360,   177,   149,   323,   633,   116,   129,
    1846,   150,   129,  1849,   129,  1518,   183,  1520,   763,  1522,
     126,  1524,   149,  1184,   191,   129,   315,   318,   129,   318,
     905,   137,   809,   116,    15,   129,   519,   129,   521,  2109,
     116,   208,   129,   282,   919,   528,   529,   530,   149,   674,
     588,   534,   535,   536,   537,   172,   539,   540,   116,   165,
     692,   724,   129,   257,   690,   129,   247,  1436,   129,   209,
     257,    15,   697,   818,   647,   373,   374,   650,   116,  1602,
    1603,   379,   380,   381,   382,   116,   384,   385,   228,  1904,
     388,   751,   390,   391,   392,   393,     8,   129,   396,  1294,
       8,   399,   969,   401,  1644,     8,  1903,   257,   406,   799,
    1295,   409,   257,   411,  1901,   413,   257,    20,   257,   417,
     418,   419,   420,  1852,   305,   306,   307,  1818,   426,  1818,
     428,   429,  1007,   431,     8,  2296,  1818,  1012,     8,   437,
     438,   257,  1818,  1979,  1818,  1818,   444,   445,   811,   257,
    1335,   449,   258,  1818,  1428,   982,   637,   984,   456,  1182,
     641,   459,  1171,  1086,   991,  1183,   637,   648,   417,   794,
     641,     8,   653,   471,   257,  1449,   474,   648,  1053,   489,
     840,   257,   480,   481,   482,   483,   257,   485,   486,   356,
     357,  1078,    15,    15,   492,   493,   494,   495,   496,   257,
     498,   254,    94,   501,   116,   503,   504,   246,   116,   507,
       8,     8,   510,   116,   839,   998,   514,   842,     8,   257,
    1805,    92,  2058,  1408,  1103,   798,   257,   800,  1805,     8,
      20,  1805,     8,  1818,   544,   191,     8,  1485,   548,  1824,
       8,  1818,   116,  1828,  1818,   543,   915,  1824,   998,     8,
    1824,     5,  1592,  1593,  1368,  1369,  1370,   129,    37,   569,
       3,    20,   560,     7,  2525,  2526,   576,  2484,     7,     8,
     568,    15,    15,   571,   572,   573,   574,   149,   149,   116,
    1842,   579,   580,    37,  2013,   583,   584,   585,   586,   587,
     916,   565,   566,   591,   592,   985,   986,   987,   988,   989,
       8,   148,  1805,   577,  2102,  2120,   129,  2102,  2144,  2021,
    2002,     8,   769,  2474,  2111,  1818,     8,     8,   799,   116,
    1103,  1824,    23,    15,  1151,  1828,   116,     8,   799,   812,
     201,   266,   288,   180,   759,  2395,    37,   963,   230,    20,
     116,   147,   148,   299,   116,   257,  1905,  1906,   116,   257,
       8,   128,   287,  1103,   257,   665,    95,   116,     8,   657,
     658,   659,   660,   661,   662,   663,     8,     3,   666,  2530,
      16,    17,  1895,   683,   180,     5,   674,  1895,     7,     9,
     157,     8,  1900,   257,    23,   124,     8,    43,   191,     5,
      26,     8,   284,     9,   692,   876,   877,   695,    37,   697,
    1275,    23,     5,   295,    21,   208,   887,    37,   116,  1035,
      37,    67,    15,     8,  1548,    37,   128,  1548,  1293,  1045,
     257,    37,   720,  1526,  1527,  1528,    82,  1530,  1531,  1308,
       3,    87,    88,   743,     5,   116,     7,   129,    94,  2542,
     217,  1450,  1451,     8,  1434,   157,    92,   745,    21,   747,
     760,   191,   750,  2556,  2269,  1640,   195,     3,   116,     8,
     257,     3,     8,   202,   774,   763,     3,   257,   208,   147,
     148,   781,    63,     8,   116,   773,    18,    68,    15,   282,
       8,   257,   285,     8,   782,   257,   784,   290,     8,   257,
     129,   789,   244,   791,  1679,   298,   794,     8,   257,    90,
      91,    21,   180,   149,   985,   986,   987,   988,   989,   807,
      21,     8,   993,   994,   985,   986,   987,   988,   989,     8,
     818,   231,   232,  2359,    21,  1308,   824,   237,  1355,  1356,
    1357,  1358,  1359,  1647,  1648,  1649,     3,   847,     8,  1020,
       8,   839,   282,     8,   842,   285,     8,  1422,    15,   257,
     848,    21,  2371,   144,     8,   201,    21,     5,  1308,    21,
    1310,     9,  1043,   154,  2123,  2124,    15,    21,    16,    17,
     880,  1054,   228,    89,  2410,    23,   257,     8,   330,   889,
     890,   891,  1065,     8,   894,  1368,  1369,  1370,     8,    37,
      21,  2109,  2321,   345,   346,   347,    21,   895,   896,   257,
    1081,    21,    15,   259,  2580,  2581,   122,  1267,    18,   200,
       5,     8,     8,   911,     9,   257,    26,  2593,  1368,  1369,
    1370,    16,    17,   897,   934,    21,    21,   338,    23,    63,
     341,   342,   343,     8,    68,   933,   314,   315,   912,   155,
     318,   319,    37,   354,   355,     8,    21,   303,     8,     8,
       8,     7,     8,  1848,     5,     8,    90,    91,     9,     3,
       8,    21,    21,    21,  2393,    16,    17,     8,    21,   185,
      21,    15,    23,  1298,     8,  1158,     8,    22,     8,    24,
    1890,  1891,  1892,  1893,  1894,     8,    37,    21,    33,    21,
      35,    21,     3,     8,  2392,  1890,  1891,  1892,  1893,  1894,
       8,     8,   218,     8,    15,  2267,    21,  1017,  1344,  1192,
     144,  1845,     8,    21,    21,  1198,  1342,  1878,  1016,  1202,
     154,  1204,  1348,  1857,  1022,  1208,  1857,  1210,     3,  1429,
    1028,  1431,     8,  1855,  1856,     8,   252,  1727,     8,  1924,
      15,  1863,  1864,  1041,  1753,    21,     5,     8,    21,     7,
       9,    21,  1388,  2482,     8,  1940,     8,    16,    17,     8,
      21,  1946,    21,  1948,    23,  1401,   200,    21,  1066,    21,
    2139,     8,  1639,  1071,    12,  1417,  1086,  1419,    37,     8,
      18,  1656,  2480,    15,  1094,     8,     3,     8,  1414,  1087,
      28,     8,    21,    31,    63,  1427,    34,   231,    21,    68,
      21,    15,     8,    41,  1102,     8,  1940,    45,    15,  1940,
     244,    49,  1946,   158,   159,    21,   161,   162,    21,    15,
       5,    90,    91,   149,     9,     8,     7,     8,  1138,    67,
      41,    16,    17,    71,    72,    15,  1460,   149,    21,    77,
      78,     8,  1851,  1718,  1853,    83,    84,   192,    86,    87,
      88,    89,    37,    91,    21,   124,    15,     8,    15,     8,
      71,     8,   100,     8,  1647,  1648,  1649,   105,     8,   107,
      21,  1181,    21,   111,    21,   144,    21,   115,     8,   117,
       8,    21,  1363,  1507,     7,   154,   124,     8,     8,   234,
     128,    21,     8,    21,   132,     8,   134,  1647,  1648,  1649,
      21,    21,   247,   141,   142,    21,     5,   145,   146,    15,
       9,    15,     8,   151,  1224,   153,     8,    16,    17,   157,
    1546,     8,     8,     8,   135,    21,  1236,  1237,  1238,    21,
       7,   200,     8,    15,    21,    21,    21,     8,    37,    16,
      17,  1424,    15,     8,     8,    21,   184,     8,   186,     8,
      21,     8,  1262,  1643,     8,   193,    21,    21,   169,    84,
      21,     8,    21,     8,   233,     8,     8,    21,   256,   679,
    1268,  1269,     8,   261,    21,   244,    21,    15,    21,    21,
     191,  1767,    15,  1769,  1770,    21,  1467,    15,     8,     8,
    1473,    16,    17,    15,  1292,  1781,    21,   208,   209,    15,
    1298,    21,    21,     8,  1302,  1791,    84,  1793,  1634,  2378,
     221,     8,   223,   224,     8,   226,    21,    15,   229,   858,
     859,     8,   861,    37,    21,    15,     8,    21,     3,  1510,
    1513,    15,  1658,  1516,    21,  1922,  1519,    15,  1521,    21,
    1523,  2315,  1525,     8,  1670,  1526,  1527,  1528,  1674,  1530,
    1531,     8,    15,  1534,    15,     8,    21,  1689,   124,     8,
    2125,  2126,  2127,    22,    21,    24,   191,  1548,    21,     8,
      15,   282,    21,     5,    33,  1373,    35,     9,    10,  2264,
      15,  2215,    21,   208,    16,    17,  1647,  1648,  1649,    15,
      22,    23,    24,    25,    26,     8,    28,    29,    30,    31,
      32,    33,  1929,    35,  2289,    37,    38,  2241,    21,   320,
    2241,   322,     8,   191,     8,     8,     7,     8,  1416,  1417,
      15,  1419,     8,    15,     8,    21,     8,    21,    21,  1427,
     208,  1429,     8,  1431,     8,    21,  1434,    21,    15,    21,
     265,     8,    15,     8,    15,    21,   856,    21,     8,    15,
     860,   251,     8,    15,    21,   865,    21,   282,   868,   869,
     285,    21,  1643,  1644,   289,    21,   404,     8,  1466,     8,
     129,     8,  1643,  1483,   299,    15,     8,  1475,     7,   304,
      21,    15,    21,  1481,    21,    15,     5,   265,     7,    21,
     124,    10,  1502,  1491,  1504,    14,    15,    16,    17,   158,
     159,    15,   161,   162,   282,     8,   246,   285,    27,   239,
    1508,   289,   290,    21,   924,    34,   926,    36,    21,  1529,
     298,   299,  1703,  1704,  1705,   246,   304,    15,    16,    17,
    1540,     5,    26,   192,   351,   352,    10,   316,   317,    27,
      14,    15,    16,    17,     8,    64,  1556,    15,    36,  2523,
       7,  1561,   240,    27,   348,   349,    16,    17,     8,  1569,
      34,    15,    36,  1602,  1603,   993,   994,  1577,   247,   248,
     508,  2576,  2577,    92,    15,   234,    64,  1587,    15,  1589,
    2457,  2458,   866,   867,  1582,     8,    15,    15,   247,     8,
      64,   149,   175,    15,  1777,     8,     8,    15,     8,  1782,
       8,    26,     7,    62,    92,  1941,    15,     8,  2317,    15,
     129,    15,  1795,     8,  1612,   287,    15,     8,    92,    21,
    1944,   314,     8,     8,   213,     5,  2213,  1625,    15,     9,
     149,     8,     8,    64,  1817,    15,    16,    17,     9,   129,
    1823,   129,    21,     9,  1827,     8,  1644,    27,     8,  1647,
    1648,  1649,   452,     8,    15,   129,    36,    15,    15,  1657,
      67,   149,    37,    15,  1845,  1846,    15,    15,  1849,    15,
    2013,    15,     8,     8,   129,   149,  1857,   203,   204,   205,
     206,   207,   201,  2392,    64,  2394,  2395,   487,     5,    26,
      15,  1689,   166,     7,     7,    21,    18,   497,    15,    16,
      17,    44,    21,    21,    37,    26,     7,    15,     7,    15,
      27,  1709,    92,   201,     5,     8,    26,    15,     9,    36,
       8,     8,     5,    15,    21,    16,    17,   201,     8,  1727,
      15,    22,    23,    15,    25,    21,    15,    28,    29,    30,
      31,    32,    15,    15,    87,  2454,    37,    64,    15,   129,
     550,   551,   552,   553,   554,  2190,    21,    21,    15,  1940,
      15,    21,    62,  1761,  1762,  1946,     8,     7,    15,   149,
    2479,  2480,     8,     8,  2100,    92,   119,     8,     8,   255,
       7,    15,   166,  1193,  2110,     8,   166,    15,  1786,  1199,
     239,    16,    21,  1203,    15,  1205,  1206,  1207,  1979,  1209,
     143,  1211,   145,    15,    15,   208,    21,     7,   191,   609,
     610,   611,   129,   613,    15,   330,   315,    15,   331,    15,
     620,   201,    15,  2532,    15,    15,    15,  2010,    15,    15,
      15,    15,   149,    15,   634,   178,    15,  1847,    15,    15,
     183,  2024,    15,    15,   187,    15,  2029,    15,   191,  2032,
       7,   194,  1862,   246,   246,     5,  1866,  2306,  2307,  2042,
       8,  2044,    15,  2046,   246,  2048,    15,    21,   246,    21,
       5,     8,   246,   216,     9,    10,   676,  2058,     7,    14,
      15,    16,    17,    21,   201,    21,     8,   230,    21,    15,
      21,   691,    27,   236,    26,    15,   173,     8,    15,    34,
      15,    36,   255,     8,     7,  1903,  1904,  1905,  1906,    21,
      71,    15,   208,     5,    15,     8,    18,     9,  1916,   719,
      15,    15,    21,    15,    16,    17,   726,    15,    15,    64,
      15,    21,   732,   733,    15,    27,    15,    21,    21,    21,
     149,    21,     5,    15,    36,  1943,     9,  2392,  2393,  1959,
    2286,    15,    21,    16,    17,    21,     8,    92,    15,     5,
      23,    15,    25,  2144,     7,    28,    29,    30,    31,    32,
       9,  1981,    64,    20,    37,   913,     5,    26,   778,    21,
      21,    21,    15,    21,    21,    21,    15,    16,    17,    15,
       8,    26,  1402,    21,   129,    26,    26,    21,    27,  1997,
      92,    21,     8,  2001,     7,  2003,    21,    36,    15,    26,
       8,   254,    21,    15,   149,  2013,    21,  2027,    15,     5,
       8,    21,  2020,  2021,    10,   132,  2036,     7,    14,    15,
      16,    17,     7,    37,  2215,    64,    21,   129,    21,    15,
       7,    27,    15,     7,  2054,   255,   353,    21,    34,    21,
      36,   851,   852,    15,    16,    17,    15,   149,    15,    15,
    2241,    15,    15,    92,    15,    27,   201,  1477,    15,    15,
      15,    15,     7,    21,    36,  1013,     8,     8,    64,     5,
       8,     7,     7,    21,    10,     8,     7,    21,    14,    15,
      16,    17,     8,    15,     7,     7,    15,    59,   352,    78,
     129,    27,    64,    21,     8,   329,    92,   351,    34,   201,
      36,     7,     7,  2111,    21,    15,    15,    15,    15,    21,
     149,     8,  2120,    17,  2122,  2123,  2124,  2125,  2126,  2127,
      92,    17,    15,     7,     7,    15,     7,   128,    64,  1269,
     896,  1999,  2152,   129,   944,   945,   946,   947,   948,   949,
     950,   951,   952,   953,   954,   955,   956,   957,   958,   959,
     960,   961,   962,   149,   222,  1875,    92,  1587,  1185,   969,
    1090,   582,   201,  1591,   974,  2264,  1876,  2358,  2359,     5,
     166,   981,   710,   983,    10,  1300,  1621,   149,    14,    15,
      16,    17,  2190,  2191,   590,   593,   933,  2269,   998,  2122,
    1903,    27,   933,   129,  1895,  2102,  1491,  1154,    34,  1728,
      36,  2172,  1997,  1475,    40,   201,  2191,  1026,  1411,  1046,
    1088,  1390,  2232,   149,  1024,  1700,    75,  1924,  1702,  2410,
    1030,  2582,  1946,   711,   246,   246,  1397,   463,    64,   201,
    1434,  2251,   792,  1643,  1035,   246,  1046,     5,   998,  1071,
     246,  1640,    10,   838,  1102,  1174,    14,    15,    16,    17,
     657,   661,  2316,  1413,  1415,   659,    92,  2151,  2530,    27,
     660,  2269,  2001,   662,  2378,   201,    34,  1016,    36,  1014,
     807,  2502,  1087,     5,  2465,  1943,  1041,  1028,    10,  2437,
    1983,   442,    14,    15,    16,    17,   788,  2391,  2532,  2558,
    2578,  2580,  2324,   129,  1753,    27,    64,  1489,  2306,  2307,
    2004,  1451,    34,   761,    36,   666,   504,   382,  2316,  1948,
    2008,  1540,  1805,   149,   415,  2296,  1828,  1780,  1824,    45,
      46,  1131,    48,    49,    92,  1785,  1818,    53,  1276,  2034,
    2347,  2033,    64,   878,  2364,    61,  1224,  2054,  1556,  1149,
      66,  2215,  1561,    69,  1569,  1577,  2251,    -1,    -1,    -1,
      -1,    77,    -1,    79,    -1,    81,    -1,    -1,    -1,    -1,
      92,   129,  1172,  1173,  1174,   201,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,  2382,    -1,    -1,    -1,   104,    -1,
     434,   149,    -1,    -1,  2392,  2393,  2406,    -1,    -1,   115,
      -1,    -1,    -1,    -1,    -1,  2415,    -1,   129,    -1,    -1,
      -1,  2421,    -1,  2423,    -1,    -1,    -1,    -1,    -1,   463,
      -1,    45,    46,   139,    48,    49,   470,   149,  2426,    53,
      -1,    -1,    -1,    -1,   478,    -1,   152,    61,    -1,  1239,
      -1,    -1,    66,   201,    -1,    69,    -1,  2457,  2458,    -1,
      -1,    -1,    -1,    77,   170,    79,     7,    81,    -1,    10,
     176,    -1,    -1,    14,    15,    16,    17,    -1,   184,    -1,
     186,    -1,   188,    -1,   190,    -1,    27,    -1,    -1,   201,
     104,   197,    -1,    34,    -1,    36,    -1,    -1,    -1,    -1,
      -1,   115,    -1,    -1,    -1,  1295,    -1,    -1,   214,   215,
       5,    -1,    -1,    -1,  2502,    10,  2504,  1307,    -1,    14,
      15,    16,    17,    64,    -1,   139,   232,    -1,    -1,    -1,
      -1,    -1,    27,   239,    -1,    -1,    -1,    -1,   152,    34,
      -1,    36,    -1,    -1,    -1,  1335,    -1,   253,    -1,    -1,
     256,    92,    -1,    -1,    -1,   589,   170,  1347,    -1,    -1,
    2560,    -1,   176,    -1,    -1,  2565,    -1,    -1,    -1,    64,
     184,    -1,   186,    -1,   188,    -1,   190,    -1,  2578,    -1,
    2580,  2581,    -1,   197,    -1,    -1,    -1,    -1,   129,    -1,
      -1,    -1,    -1,  2593,    -1,    -1,    -1,    92,    -1,    -1,
     214,   215,    -1,    -1,     5,    -1,  1396,  1397,   149,    10,
      -1,    -1,    -1,    14,    15,    16,    17,    -1,   232,   653,
      -1,  1411,    -1,    -1,    -1,   166,    27,    -1,  1418,    -1,
      -1,    -1,    -1,    34,   129,    36,    -1,    -1,  1428,   253,
      -1,    -1,   256,    -1,  1434,    -1,  1436,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   149,    -1,    -1,    -1,    -1,  1449,
     201,     7,    -1,    64,    10,    -1,    -1,    -1,    14,    15,
      16,    17,    -1,  1463,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    27,    -1,    -1,  1474,    -1,    -1,    -1,    34,     5,
      36,    92,    -1,     9,    -1,    -1,    -1,  1487,    -1,    -1,
      16,    17,  1492,    -1,  1494,    -1,   201,    23,    -1,    25,
      -1,   745,    28,    29,    30,    31,    32,    -1,    64,    -1,
      -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,   129,   763,
      -1,    -1,    78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   777,    -1,    -1,    92,    -1,   149,    -1,
      -1,   785,   786,    -1,    -1,    -1,   790,    -1,    -1,     7,
      -1,    -1,    10,    -1,    -1,    -1,    14,    15,    16,    17,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    27,
      -1,    -1,    -1,   129,   818,    -1,    34,    -1,    36,    -1,
      -1,    -1,   826,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     201,    -1,    -1,   149,    -1,    -1,    -1,  1597,    -1,    -1,
       7,    -1,    -1,    10,    -1,    -1,    64,    14,    15,    16,
      17,   855,    -1,   857,    -1,    -1,    -1,    -1,    -1,    -1,
      27,    -1,    -1,    -1,    -1,    -1,    -1,    34,    -1,    36,
      -1,    -1,   876,   877,    92,    -1,    -1,    -1,  1638,  1639,
    1640,    -1,    -1,   887,   888,   201,     7,    -1,    -1,    10,
      -1,    -1,    -1,    14,    15,    16,    17,    64,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    27,    -1,    -1,  1669,
      -1,   129,  1672,    34,    -1,    36,    -1,    -1,    -1,     7,
    1680,    -1,    10,    -1,    -1,    92,    14,    15,    16,    17,
      -1,   149,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    27,
      -1,    -1,    -1,    64,    -1,    -1,    34,    -1,    36,    -1,
       7,    -1,  1712,    10,  1714,    -1,    -1,    14,    15,    16,
      17,    -1,   129,   967,   968,     7,    -1,    -1,    10,    44,
      27,    92,    14,    15,    16,    17,    64,    34,    -1,    36,
      -1,    -1,   149,   201,    -1,    27,    -1,    -1,    -1,   993,
     994,    -1,    34,    -1,    36,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    92,    -1,    -1,    64,   129,    -1,
      -1,    -1,    87,    -1,    -1,    -1,  1020,    -1,    93,    -1,
      -1,    -1,    64,    -1,    -1,    -1,    -1,    -1,   149,    -1,
      -1,    -1,    -1,    -1,   201,    92,    -1,    -1,    -1,  1043,
      -1,   129,    -1,    -1,   119,    -1,    -1,    10,    -1,    -1,
      92,    14,    15,    16,    17,    -1,    -1,    -1,    -1,    -1,
      -1,   149,    -1,    -1,    27,    -1,    -1,    -1,   143,    -1,
     145,    34,   129,    36,    -1,    -1,    -1,  1081,    10,    -1,
     201,    -1,    14,    15,    16,    17,    -1,   129,    -1,    -1,
      -1,    -1,   149,    -1,    -1,    27,    -1,    -1,  1858,    -1,
      -1,    64,    34,   178,    36,  1865,    -1,   149,   183,  1869,
    1870,    -1,   187,   201,    -1,    -1,   191,  1877,    -1,   194,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,
      -1,    -1,    64,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   216,    -1,    -1,   201,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   230,    -1,    -1,  1918,   201,
      92,   236,    -1,    -1,  1924,    -1,   129,  1171,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     5,    -1,     7,    -1,
       9,    10,    -1,    -1,    -1,    -1,   149,    16,    17,  1949,
      -1,    -1,    21,    22,    23,    24,    25,   129,  1958,    28,
      29,    30,    31,    32,    33,    -1,    35,     5,    37,    38,
      -1,     9,    10,    -1,    -1,  1975,  1976,   149,    16,    17,
      -1,    -1,    -1,    -1,    22,    23,    24,    25,    -1,    -1,
      28,    29,    30,    31,    32,    33,    34,    35,   201,    37,
      38,    -1,    -1,    -1,  2004,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,  2012,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,  2022,    -1,    -1,    -1,    -1,    -1,    -1,   201,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,  2038,     5,
    2040,     7,    -1,     9,    10,    -1,    -1,    -1,    -1,    -1,
      16,    17,    -1,    -1,    -1,  1299,    22,    23,    24,    25,
      -1,    -1,    28,    29,    30,    31,    32,    33,    -1,    35,
       5,    37,    38,     8,     9,    10,    -1,    -1,    -1,    -1,
      -1,    16,    17,    -1,    -1,    -1,    -1,    22,    23,    24,
      25,    -1,    -1,    28,    29,    30,    31,    32,    33,    -1,
      35,   621,    37,    38,    -1,     5,    -1,    -1,     8,     9,
      10,    -1,    -1,    -1,    -1,    -1,    16,    17,    -1,  1363,
      -1,    -1,    22,    23,    24,    25,    -1,    -1,    28,    29,
      30,    31,    32,    33,  2134,    35,    -1,    37,    38,  2139,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     5,
    2150,  2151,     8,     9,    10,  2155,    -1,    -1,    -1,    -1,
      16,    17,    -1,    -1,    -1,  2165,    22,    23,    24,    25,
      -1,  2171,    28,    29,    30,    31,    32,    33,    -1,    35,
      -1,    37,    38,    -1,    -1,  1429,    -1,  1431,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,  1450,  1451,    -1,     5,
      -1,    -1,    -1,     9,  2214,    -1,    -1,  1461,    -1,    -1,
      16,    17,    -1,  1467,    -1,  1469,    22,    23,  1472,    25,
      -1,    -1,    28,    29,    30,    31,    32,    -1,    -1,    35,
    1484,    37,    38,  2243,  1488,  1489,    -1,    -1,    -1,    -1,
      -1,   771,    -1,    -1,    -1,   775,    -1,    -1,  2258,    -1,
      -1,    -1,    -1,    -1,  2264,    -1,  1510,    -1,    -1,    -1,
      -1,    -1,    -1,  1517,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,  1526,  1527,  1528,    -1,  1530,  1531,    -1,    -1,
    1534,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    2300,     5,   822,    -1,  1548,     9,    10,    -1,    -1,    -1,
      -1,    -1,    16,    17,   834,  2315,    -1,    21,    22,    23,
      24,    25,    -1,    -1,    28,    29,    30,    31,    32,    33,
      -1,    35,    -1,    37,    38,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   881,   882,   883,   884,    -1,   886,    -1,  1612,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,  2377,  2378,    -1,
      -1,  2381,    -1,  2383,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,  2398,  2399,
    1644,  2401,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,  2411,    -1,    -1,    -1,    -1,    -1,    -1,   938,  2419,
     940,   941,   942,   943,     5,    -1,    -1,    -1,     9,    10,
      -1,    -1,    -1,    -1,    -1,    16,    17,  2437,    -1,    -1,
      21,    22,    23,    24,    25,    -1,    -1,    28,    29,    30,
      31,    32,    33,    -1,    35,    -1,    37,    38,    -1,  1703,
    1704,  1705,    -1,    -1,    -1,  2465,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,  2473,  2474,    -1,    -1,    -1,    -1,   999,
      -1,    -1,    -1,    -1,  2484,    -1,    -1,  2487,    -1,    -1,
      -1,  2491,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,  1021,    -1,    -1,    -1,    -1,    -1,    -1,    -1,  1753,
      -1,  1031,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,  2523,    -1,    -1,    -1,    -1,    -1,    -1,
    2530,    -1,  1052,    -1,  2534,    -1,    -1,    -1,  2538,    -1,
      -1,    -1,  2542,    -1,    -1,    -1,    -1,    -1,  1068,    -1,
      -1,    -1,  1072,  1073,    -1,    -1,  2556,  2557,  2558,     5,
      -1,     7,    -1,     9,    10,    -1,    -1,    -1,    -1,    -1,
      16,    17,    -1,    -1,    -1,    -1,    22,    23,    24,    25,
      -1,    -1,    28,    29,    30,    31,    32,    33,    -1,    35,
      -1,    37,    38,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,  1845,  1846,    -1,    -1,  1849,    -1,  1851,  1852,  1853,
    1130,  1855,  1856,  1857,    -1,    -1,    -1,    -1,    -1,  1863,
    1864,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,  1890,  1891,  1892,  1893,
    1894,    -1,    -1,     5,    -1,     7,    -1,     9,    10,    -1,
      -1,    -1,    -1,    -1,    16,    17,    -1,    -1,  1188,  1189,
      22,    23,    24,    25,    -1,    -1,    28,    29,    30,    31,
      32,    33,    -1,    35,    -1,    37,    38,     5,    -1,    -1,
      -1,     9,    10,    -1,    -1,    -1,  1940,    -1,    16,    17,
      -1,    -1,  1946,    21,    22,    23,    24,    25,    -1,    -1,
      28,    29,    30,    31,    32,    33,    -1,    35,    -1,    37,
      38,     5,    -1,    -1,    -1,     9,    10,    -1,    -1,    -1,
      -1,    -1,    16,    17,    -1,  1979,  1980,    21,    22,    23,
      24,    25,    -1,    -1,    28,    29,    30,    31,    32,    33,
      -1,    35,    -1,    37,    38,     5,    -1,    -1,  2002,     9,
      10,    -1,    -1,    -1,    -1,    -1,    16,    17,    -1,  2013,
      -1,    21,    22,    23,    24,    25,    -1,    -1,    28,    29,
      30,    31,    32,    33,    -1,    35,    -1,    37,    38,    -1,
    2034,  1311,  1312,  1313,  1314,  1315,  1316,  1317,  1318,  1319,
    1320,  1321,  1322,  1323,  1324,  1325,  1326,  1327,  1328,  1329,
      -1,    -1,    -1,    -1,  2058,  2059,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,  1345,    -1,    -1,    -1,    -1,
      -1,     5,  1352,    -1,  1354,     9,    10,    -1,    -1,    -1,
      -1,    -1,    16,    17,  1364,    -1,    -1,    21,    22,    23,
      24,    25,    -1,    -1,    28,    29,    30,    31,    32,    33,
      -1,    35,    -1,    37,    38,    -1,     5,  2111,    -1,    -1,
       9,    10,    -1,    -1,    -1,  1395,    -1,    16,    17,    -1,
      -1,    -1,    21,    22,    23,    24,    25,    -1,    -1,    28,
      29,    30,    31,    32,    33,    -1,    35,  2141,    37,    38,
    2144,     5,  2146,    -1,    -1,     9,    10,    -1,    -1,    -1,
      -1,    -1,    16,    17,    -1,    -1,    -1,    21,    22,    23,
      24,    25,    -1,    -1,    28,    29,    30,    31,    32,    33,
      -1,    35,    -1,    37,    38,    -1,     5,    -1,    -1,    -1,
       9,    10,    -1,    -1,  1464,    -1,    -1,    16,    17,    -1,
      -1,    -1,    21,    22,    23,    24,    25,    -1,  1478,    28,
      29,    30,    31,    32,    33,    -1,    35,    -1,    37,    38,
      -1,  2215,    -1,    -1,    -1,    -1,     5,    -1,    -1,  1499,
       9,    10,    -1,    -1,    -1,    -1,    -1,    16,    17,    -1,
      -1,    -1,    21,    22,    23,    24,    25,  2241,    -1,    28,
      29,    30,    31,    32,    33,    -1,    35,    -1,    37,    38,
       5,    -1,     7,    -1,     9,    10,    -1,    -1,    -1,    -1,
      -1,    16,    17,    -1,    -1,  1545,    -1,    22,    23,    24,
      25,    -1,    -1,    28,    29,    30,    31,    32,    33,    -1,
      35,    -1,    37,    38,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     5,    -1,    -1,    -1,     9,    10,    -1,
      -1,    -1,    -1,  2317,    16,    17,  1596,  2321,    -1,    21,
      22,    23,    24,    25,    -1,    -1,    28,    29,    30,    31,
      32,    33,    -1,    35,    -1,    37,    38,    -1,    -1,    -1,
       5,    -1,     7,    -1,     9,    10,    -1,    -1,    -1,  1629,
      -1,    16,    17,    -1,  2358,  2359,  1636,    22,    23,    24,
      25,    -1,  1642,    28,    29,    30,    31,    32,    33,    -1,
      35,    -1,    37,    38,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,  1665,    -1,    -1,  2392,  2393,
    2394,  2395,    -1,    -1,    -1,    -1,  2400,  1677,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,  1685,  2410,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,  1695,    -1,    -1,  1698,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,  1713,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,  1725,    -1,    -1,    -1,    -1,
    2454,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,  1739,
      -1,    -1,    -1,    -1,    -1,  1745,    -1,    -1,    -1,    -1,
      -1,  1751,    -1,    -1,    -1,  2479,  2480,     5,  2482,    -1,
      -1,     9,    10,    -1,    -1,  2489,    -1,    -1,    16,    17,
      -1,    -1,    -1,    21,    22,    23,    24,    25,    -1,    -1,
      28,    29,    30,    31,    32,    33,    -1,    35,    -1,    37,
      38,    -1,     5,    -1,     7,    -1,     9,    10,    -1,    -1,
      -1,    -1,    -1,    16,    17,    -1,    -1,    -1,  2532,    22,
      23,    24,    25,    -1,    -1,    28,    29,    30,    31,    32,
      33,    -1,    35,    -1,    37,    38,     5,    -1,     7,    -1,
       9,    10,    -1,    -1,    -1,    -1,    -1,    16,    17,    -1,
      -1,    -1,    -1,    22,    23,    24,    25,    -1,    -1,    28,
      29,    30,    31,    32,    33,    -1,    35,    -1,    37,    38,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      14,    15,    16,    17,    -1,    19,    -1,    21,    -1,    23,
      24,  1881,    26,    -1,    -1,    -1,    30,    -1,    32,    33,
      -1,    -1,    36,    37,    38,    -1,    40,    -1,    42,    43,
      -1,    -1,    46,    -1,    -1,    -1,    -1,    51,    52,    53,
      54,    55,    56,    57,    58,    -1,    60,    61,    -1,    -1,
      64,    65,    66,    -1,    68,  1925,    70,    -1,    -1,    73,
      74,    -1,    -1,    -1,    -1,    -1,    -1,    81,    82,    -1,
      -1,    85,    -1,    -1,    -1,  1945,    -1,  1947,    92,    -1,
      94,  1951,    96,    97,    -1,    -1,    -1,   101,   102,   103,
      -1,    -1,   106,    -1,    -1,   109,   110,    -1,   112,   113,
     114,    -1,    -1,    -1,  1974,   119,   120,  1977,    -1,   123,
      -1,   125,   126,   127,    -1,   129,   130,    -1,    -1,   133,
      -1,   135,   136,   137,   138,   139,   140,    -1,    -1,  1999,
      -1,    -1,    -1,   147,   148,   149,    -1,    -1,    -1,    -1,
     154,   155,   156,    -1,   158,   159,   160,   161,   162,   163,
     164,   165,   166,   167,   168,   169,   170,   171,   172,   173,
     174,   175,   176,   177,   178,   179,   180,   181,   182,   183,
      -1,   185,    -1,   187,    -1,   189,    -1,   191,   192,    -1,
     194,   195,   196,   197,   198,   199,   200,   201,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     5,  2077,    -1,     8,
       9,    10,    -1,    -1,    -1,    -1,  2086,    16,    17,    -1,
    2090,  2091,    -1,    22,    23,    24,    25,    -1,  2098,    28,
      29,    30,    31,    32,    33,    -1,    35,     5,    37,    38,
     254,     9,    10,   257,    -1,   259,    -1,    -1,    16,    17,
      -1,    -1,    -1,    21,    22,    23,    24,    25,    -1,  2129,
      28,    29,    30,    31,    32,    33,    -1,    35,    -1,    37,
      38,    -1,    -1,    -1,    -1,    -1,    -1,    -1,  2148,    -1,
       5,    -1,    -1,  2153,     9,    10,    -1,    -1,    -1,    -1,
      -1,    16,    17,  2163,  2164,    -1,    21,    22,    23,    24,
      25,    -1,    -1,    28,    29,    30,    31,    32,    33,    -1,
      35,    -1,    37,    38,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     5,    -1,    -1,  2194,     9,    10,    -1,    -1,    -1,
      -1,    -1,    16,    17,  2204,  2205,    -1,    21,    22,    23,
      24,    25,    -1,    -1,    28,    29,    30,    31,    32,    33,
      -1,    35,     5,    37,    38,     8,     9,    10,    -1,    -1,
      -1,    -1,    -1,    16,    17,    -1,    -1,    -1,    -1,    22,
      23,    24,    25,    -1,    -1,    28,    29,    30,    31,    32,
      33,    -1,    35,    -1,    37,    38,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,  2279,
      -1,  2281,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     5,
    2290,    -1,    -1,     9,    10,  2295,    -1,    -1,  2298,    -1,
      16,    17,    -1,    -1,    -1,    21,    22,    23,    24,    25,
      -1,  2311,    28,    29,    30,    31,    32,    33,    -1,    35,
       5,    37,    38,    -1,     9,    10,    -1,    -1,    -1,    -1,
      -1,    16,    17,    -1,    -1,    -1,    21,    22,    23,    24,
      25,    -1,    -1,    28,    29,    30,    31,    32,    33,    -1,
      35,    -1,    37,    38,    -1,    -1,  2356,    -1,    -1,    -1,
      -1,    -1,  2362,    -1,    -1,     5,    -1,    -1,  2368,     9,
      10,    -1,    -1,    -1,    -1,    -1,    16,    17,    -1,    -1,
      -1,    21,    22,    23,    24,    25,    -1,  2387,    28,    29,
      30,    31,    32,    33,    -1,    35,     5,    37,    38,     8,
       9,    10,    -1,    -1,    -1,    -1,    -1,    16,    17,    -1,
      -1,    -1,    -1,    22,    23,    24,    25,    -1,    -1,    28,
      29,    30,    31,    32,    33,    -1,    35,    -1,    37,    38,
    2430,    -1,  2432,    -1,    -1,    -1,  2436,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,  2447,  2448,    -1,
      -1,    -1,  2452,     5,   598,   599,    -1,     9,    10,    -1,
      -1,    -1,  2462,    -1,    16,    17,  2466,    -1,    -1,    21,
      22,    23,    24,    25,    -1,    -1,    28,    29,    30,    31,
      32,    33,    -1,    35,    -1,    37,    38,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,  2498,    -1,
      -1,    -1,    -1,  2503,    -1,  2505,    -1,    -1,    -1,     5,
      -1,    -1,     8,     9,    10,     6,    -1,  2517,    -1,  2519,
      16,    17,    13,    -1,    -1,    -1,    22,    23,    24,    25,
      -1,    -1,    28,    29,    30,    31,    32,    33,    -1,    35,
      -1,    37,    38,    -1,    -1,    -1,    -1,  2547,    -1,    -1,
      -1,  2551,    43,    44,    45,    46,    -1,    48,    49,    50,
      51,    52,    53,  2563,    -1,    56,    57,    -1,    -1,    -1,
      61,    62,    63,    -1,    65,    66,    67,    68,    69,    70,
      -1,    72,    73,    74,    75,    76,    77,    -1,    79,    80,
      81,    82,    83,    84,    -1,    -1,    87,    88,    89,    90,
      91,    -1,    -1,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,    -1,   117,    -1,   119,   120,
      -1,   122,   123,   124,    -1,    -1,   127,    -1,    -1,   130,
     131,    -1,   133,   134,   135,   136,    -1,   138,   139,   140,
     141,   142,   143,   144,   145,   146,    -1,    -1,    -1,    -1,
      -1,   152,   153,   154,   155,    -1,    -1,    -1,    -1,   160,
      -1,    -1,   163,   164,    -1,    -1,   167,   168,    -1,   170,
      -1,    -1,    -1,   174,    -1,   176,    -1,   178,    -1,    -1,
      -1,   182,   183,   184,   185,   186,   187,   188,   189,   190,
     191,    -1,   193,   194,   195,    -1,   197,    -1,   199,   200,
      -1,   202,    -1,   204,   205,   206,   207,    -1,    -1,   210,
     211,   212,    -1,   214,   215,   216,    -1,   218,   219,   220,
      -1,   222,    -1,   224,   225,   226,   227,   228,    -1,   230,
      -1,   232,   233,    -1,    -1,   236,   237,   238,    -1,    -1,
     241,   242,    -1,   244,   245,    -1,   247,   248,    -1,    -1,
      -1,   252,   253,    -1,    -1,   256,    -1,    -1,   259,    -1,
      -1,    -1,   263,   264,    -1,    -1,   267,   268,   269,    -1,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,    -1,   283,    -1,    -1,   286,    -1,    -1,    -1,   290,
     291,   292,   293,   294,    -1,   296,   297,    -1,    -1,   300,
     301,   302,   303,    -1,    -1,    -1,    -1,   308,   309,   310,
     311,   312,   313,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     321,    -1,    -1,   324,   325,   326,   327,   328,    -1,    -1,
      -1,   332,   333,   334,   335,   336,   337,     5,   339,   340,
       8,     9,    10,   344,    -1,    -1,    -1,    -1,    16,    17,
      -1,    -1,    -1,    -1,    22,    23,    24,    25,    -1,    -1,
      28,    29,    30,    31,    32,    33,    -1,    35,     5,    37,
      38,    -1,     9,    10,    -1,    -1,    -1,    -1,    -1,    16,
      17,    -1,    -1,    -1,    21,    22,    23,    24,    25,    -1,
      -1,    28,    29,    30,    31,    32,    33,    -1,    35,     5,
      37,    38,     8,     9,    10,    -1,    -1,    -1,    -1,    -1,
      16,    17,    -1,    -1,    -1,    -1,    22,    23,    24,    25,
      -1,    -1,    28,    29,    30,    31,    32,    33,    -1,    35,
       5,    37,    38,    -1,     9,    10,    -1,    -1,    -1,    -1,
      -1,    16,    17,    -1,    -1,    -1,    21,    22,    23,    24,
      25,    -1,    -1,    28,    29,    30,    31,    32,    33,    -1,
      35,     5,    37,    38,    -1,     9,    10,    -1,    -1,    -1,
      -1,    -1,    16,    17,    -1,    -1,    -1,    21,    22,    23,
      24,    25,    -1,    -1,    28,    29,    30,    31,    32,    33,
      -1,    35,     5,    37,    38,     8,     9,    10,    -1,    -1,
      -1,    -1,    -1,    16,    17,    -1,    -1,    -1,    -1,    22,
      23,    24,    25,    -1,    -1,    28,    29,    30,    31,    32,
      33,    -1,    35,     5,    37,    38,     8,     9,    10,    -1,
      -1,    -1,    -1,    -1,    16,    17,    -1,    -1,    -1,    -1,
      22,    23,    24,    25,    -1,    -1,    28,    29,    30,    31,
      32,    33,    -1,    35,     5,    37,    38,    -1,     9,    10,
      -1,    -1,    -1,    -1,    -1,    16,    17,    -1,    -1,    -1,
      21,    22,    23,    24,    25,    -1,    -1,    28,    29,    30,
      31,    32,    33,    -1,    35,     5,    37,    38,    -1,     9,
      10,    -1,    -1,    -1,    -1,    -1,    16,    17,    -1,    -1,
      -1,    21,    22,    23,    24,    25,    -1,    -1,    28,    29,
      30,    31,    32,    33,    -1,    35,     5,    37,    38,     8,
       9,    10,    -1,    -1,    -1,    -1,    -1,    16,    17,    -1,
      -1,    -1,    -1,    22,    23,    24,    25,    -1,    -1,    28,
      29,    30,    31,    32,    33,    -1,    35,    -1,    37,    38,
       5,    -1,     7,    -1,     9,    10,    -1,    -1,    -1,    -1,
      -1,    16,    17,    -1,    -1,    -1,    -1,    22,    23,    24,
      25,    -1,    -1,    28,    29,    30,    31,    32,    33,    -1,
      35,     5,    37,    38,    -1,     9,    10,    -1,    -1,    -1,
      -1,    -1,    16,    17,    -1,    -1,    -1,    -1,    22,    23,
      24,    25,    -1,    -1,    28,    29,    30,    31,    32,    33,
      -1,    35,     5,    37,    38,    -1,     9,    -1,    -1,    -1,
      -1,    -1,    -1,    16,    17,    -1,    -1,    -1,    -1,    22,
      23,    24,    25,    -1,    -1,    28,    29,    30,    31,    32,
      33,    -1,    35,    -1,    37,    38
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   361,     0,     1,   150,   257,   362,   363,   116,     6,
      13,    43,    44,    45,    46,    48,    49,    50,    51,    52,
      53,    56,    57,    61,    62,    63,    65,    66,    67,    68,
      69,    70,    72,    73,    74,    75,    76,    77,    79,    80,
      81,    82,    83,    84,    87,    88,    89,    90,    91,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   117,   119,   120,   122,   123,   124,   127,   130,   131,
     133,   134,   135,   136,   138,   139,   140,   141,   142,   143,
     144,   145,   146,   152,   153,   154,   155,   160,   163,   164,
     167,   168,   170,   174,   176,   178,   182,   183,   184,   185,
     186,   187,   188,   189,   190,   191,   193,   194,   195,   197,
     199,   200,   202,   204,   205,   206,   207,   210,   211,   212,
     214,   215,   216,   218,   219,   220,   222,   224,   225,   226,
     227,   228,   230,   232,   233,   236,   237,   238,   241,   242,
     244,   245,   247,   248,   252,   253,   256,   259,   263,   264,
     267,   268,   269,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   283,   286,   290,   291,   292,   293,
     294,   296,   297,   300,   301,   302,   303,   308,   309,   310,
     311,   312,   313,   321,   324,   325,   326,   327,   328,   332,
     333,   334,   335,   336,   337,   339,   340,   344,   364,   366,
     369,   381,   382,   386,   387,   388,   394,   395,   396,   397,
     399,   400,   402,   404,   405,   406,   407,   414,   415,   416,
     417,   418,   419,   423,   424,   425,   429,   430,   468,   470,
     483,   526,   527,   529,   530,   536,   537,   538,   539,   546,
     547,   548,   549,   551,   554,   558,   559,   560,   561,   562,
     563,   569,   570,   571,   582,   583,   584,   586,   589,   592,
     597,   598,   600,   602,   604,   607,   608,   632,   633,   644,
     645,   646,   647,   652,   655,   658,   661,   662,   712,   713,
     714,   715,   716,   717,   718,   719,   725,   727,   729,   731,
     733,   734,   735,   736,   737,   740,   742,   743,   744,   747,
     748,   752,   753,   755,   756,   757,   758,   759,   760,   761,
     764,   769,   774,   776,   777,   778,   779,   781,   782,   783,
     784,   785,   786,   803,   806,   807,   808,   809,   815,   818,
     823,   824,   825,   828,   829,   830,   831,   832,   833,   834,
     835,   836,   837,   838,   839,   840,   841,   842,   847,   848,
     849,   850,   851,   861,   862,   863,   865,   866,   867,   868,
     869,   874,   894,    15,   493,   493,   555,   555,   555,   555,
     555,   493,   555,   555,   365,   555,   555,   555,   493,   555,
     493,   555,   555,   493,   555,   555,   555,   492,   555,   493,
     555,   555,     7,    15,   494,    15,   493,   615,   555,   493,
     378,   555,   555,   555,   555,   555,   555,   555,   555,   555,
     555,   129,   371,   535,   535,   555,   555,   555,   493,   555,
     371,   555,   493,   493,   555,   555,   492,   365,   493,   493,
      64,   377,   555,   555,   493,   493,   555,   493,   493,   493,
     493,   493,   555,   432,   555,   555,   555,   371,   469,   365,
     493,   555,   555,   555,   493,   555,   493,   555,   555,   493,
     555,   555,   555,   493,   365,   493,   378,   555,   555,   378,
     555,   493,   555,   555,   555,   493,   555,   555,   493,   555,
     493,   555,   555,   555,   555,   555,   555,    15,   493,   593,
     493,   365,   493,   493,   555,   555,   555,    15,     8,   493,
     493,   555,   555,   555,   493,   555,   555,   555,   555,   555,
     555,   555,   555,   555,   555,   555,   555,   555,   555,   555,
     555,   555,   555,   555,   555,   555,   555,   555,   555,   555,
     555,   493,   555,   493,   555,   555,   555,   555,   493,   555,
     555,   555,   555,   555,   555,   555,   555,   900,   900,   900,
     900,   900,   900,   257,   581,   124,   233,   402,    15,   374,
     581,     8,     8,     8,     8,     7,     8,   124,   366,   389,
       8,   371,   403,     8,     8,     8,     8,     8,   550,     8,
     550,     8,     8,     8,     8,   550,   581,     7,   218,   252,
     527,   529,   538,   539,   239,   547,   547,    10,    14,    15,
      16,    17,    27,    34,    36,    64,    92,   149,   201,   371,
     383,   499,   500,   502,   503,   504,   505,   511,   512,   513,
     514,   515,   518,    15,   555,     5,     9,    15,    16,    17,
     129,   501,   503,   511,   565,   579,   580,   555,    15,   565,
     555,     5,   564,   565,   580,   565,     8,     8,     8,     8,
       8,     8,     8,     8,     7,     8,     8,     5,     7,   371,
     642,   643,   371,   635,   494,    15,    15,   149,   482,   371,
     371,   745,   746,     8,   371,   659,   660,   746,   371,   373,
     371,    15,   531,   577,    23,    37,   371,   421,   422,    15,
     371,   605,   371,   673,   673,   371,   656,   657,   371,   534,
     431,    15,   371,   585,   149,   751,   534,     7,   477,   478,
     493,   616,   617,   371,   611,   617,    15,   556,   371,   587,
     588,   534,    15,    15,   534,   751,   535,   534,   534,   534,
     534,   371,   534,   374,   534,    15,   426,   494,   502,   503,
      15,   368,   371,   371,   653,   654,   484,   485,   486,   487,
       8,   674,   741,    15,   371,   599,   371,   590,   591,   578,
      15,    15,   371,   494,    15,   499,   754,    15,    15,   371,
     728,   730,     8,   371,    37,   420,    15,   503,   504,   494,
      15,    15,   556,   482,   494,   503,   371,   720,     5,    15,
     579,   580,   494,   371,   372,   494,   578,    15,   502,   636,
     637,   611,   615,   371,   603,   371,   700,   700,    15,   371,
     601,   720,   499,   510,   494,   378,    15,   371,   706,   706,
     706,   706,   706,     7,   499,   594,   595,   371,   596,   494,
     367,   371,   494,   371,   726,   728,   371,   493,   494,   371,
     471,    15,    15,   578,   371,    15,   617,    15,   617,   617,
     617,   617,   789,   845,   617,   617,   617,   617,   617,   617,
     789,   371,   378,   852,   853,   854,    15,    15,   378,   864,
      15,   499,   499,   499,   499,   498,   499,    15,    15,    15,
      15,    15,   371,   892,    15,   365,   365,   124,     5,    21,
     371,   375,   376,   370,   378,   371,   371,   371,   422,     7,
     378,   365,   124,   371,   371,     5,    15,   409,   410,   371,
     422,   422,   422,   422,   421,   502,   420,   371,   371,   426,
     433,   434,   436,   437,   555,   555,   239,   412,   499,   500,
     499,   499,   499,   499,     5,     9,    16,    17,    22,    23,
      24,    25,    26,    28,    29,    30,    31,    32,    33,    35,
      37,    38,   383,    15,   246,     3,    15,   246,   246,    15,
     508,   509,    21,   552,   577,   510,     5,     9,   166,   566,
     567,   568,   579,    26,   579,     5,     9,    23,    37,   501,
     578,   579,   578,     8,    15,   503,   572,   573,    15,   499,
     500,   515,   574,   575,   576,   574,   585,   371,   599,   601,
     603,   605,   371,     7,   378,   726,     8,    21,   637,   422,
     524,   499,   240,   550,    15,   378,    15,   476,     8,   577,
       7,   499,   532,   533,   534,    15,   371,   476,   422,   481,
     482,     8,   433,   524,   476,    15,     8,    21,     5,     7,
     479,   480,   499,   371,     8,    21,     5,    58,    86,   126,
     137,   165,   258,   618,   614,   615,   175,   606,   499,   149,
     545,     8,   499,   499,   370,   371,   427,   428,   502,   507,
     371,    26,   371,   540,   541,   543,   374,     8,     8,    15,
     231,   402,   488,   378,     8,   741,   371,   502,   710,   720,
     738,   739,     8,   565,    26,     5,     9,    16,    17,    22,
      23,    24,    25,    28,    29,    30,    31,    32,    33,    34,
      35,    37,    38,   383,   384,   385,   371,   378,   392,   502,
     499,    15,   378,   371,   371,   502,   502,   525,     8,   675,
     732,   371,   502,   663,   371,   466,   467,   545,   422,    18,
     578,   579,   578,   398,   401,   642,   637,     7,   615,   617,
     710,   720,   721,   722,   421,   422,   460,   461,    62,   502,
     765,    15,    15,     7,     8,    21,   593,   422,   374,   422,
     476,     8,   672,   697,    21,   378,   371,     8,   499,   499,
     476,   502,   550,   810,   502,   287,   822,   822,   550,   819,
     822,    15,   550,   787,   550,   826,   787,   787,   550,   804,
     550,   816,   476,   147,   148,   180,   314,   315,   318,   319,
     379,   855,   856,   857,     8,    21,   503,   678,   858,    21,
     858,   379,   856,   378,   762,   763,     8,     8,     8,     8,
     502,   505,   506,   780,   663,   378,   875,   876,   877,   878,
     879,   880,   881,   378,   884,   885,   886,   887,   888,   378,
     889,   890,     8,   378,   895,   896,   371,   367,   365,     8,
      21,   213,   379,   476,    44,    87,    93,   119,   143,   145,
     178,   183,   187,   191,   194,   216,   230,   236,   390,   391,
     393,   371,   365,   493,   556,   577,   403,   476,   550,   550,
       8,    37,    15,   371,   439,   444,   378,    15,   519,    21,
       8,   499,   499,   499,   499,   499,   499,   499,   499,   499,
     499,   499,   499,   499,   499,   499,   499,   499,   499,   499,
     577,    64,   129,   495,   497,   577,   502,   513,   516,    64,
     516,   510,     8,    21,     5,   499,   553,   568,     8,    21,
       5,     9,   499,    21,   499,   579,   579,   579,   579,   579,
      21,   572,   572,     8,   499,   500,   575,   576,     8,     8,
       8,   476,   476,   493,    43,    67,    82,    87,    88,    94,
     228,   259,   303,   646,   643,   378,   506,   522,    21,   371,
      15,   498,    67,   477,   660,   499,     7,     8,    21,   552,
      37,     8,    21,   657,   502,   505,   521,   523,   577,   749,
     479,     7,   476,   617,    15,    15,    15,    15,    15,    15,
     606,   617,   371,    21,   557,   588,    21,    21,    15,     8,
      21,     8,   509,   503,     8,   542,    26,   370,   654,   485,
     129,   489,   490,   491,   407,   169,   208,   282,   378,    15,
       7,     8,    21,   591,   574,    21,    21,   147,   148,   180,
      21,    18,    21,     7,   499,   517,   175,   323,    37,     8,
      21,   378,     8,    21,    26,     8,    21,   557,   499,    21,
     462,   463,   462,    21,     7,   617,   606,    15,     7,     8,
      21,     8,    15,    15,    26,   707,   708,   710,   498,   499,
     595,   378,     8,   697,     8,   672,   403,   393,   380,    21,
      21,    21,   617,   550,    21,   617,   550,   846,   617,   550,
     617,   550,   617,   550,   617,   550,    15,    15,    15,    15,
      15,    15,   378,   854,     8,    21,    21,   182,   315,   318,
       8,    21,   378,   378,   378,   499,    15,    15,     8,    21,
      21,   183,   191,   208,   356,   357,     8,    21,    41,   209,
     228,     8,    21,   338,   341,   342,   343,   354,   355,     8,
      21,   378,   244,   330,   345,   346,   347,     8,    21,   374,
     371,   376,    15,   408,   409,   476,   493,    15,     7,     8,
     371,   476,    15,   513,     5,   411,   499,   568,   422,   502,
     436,    15,    16,    17,    27,    36,    59,    64,    92,   149,
     201,   435,   437,   447,   448,   449,   450,   451,   452,   453,
     454,   439,   444,   445,   446,    15,   440,   441,    62,   499,
     574,   500,   495,    21,     8,   496,   499,   517,   568,     7,
     577,   482,   499,   577,     8,   573,    21,     8,     8,     8,
     500,   576,   500,   576,   500,   576,   371,   255,     8,    21,
     482,   481,    21,     7,    21,   499,   532,    21,   482,   550,
       8,    21,   568,   750,     8,    21,   480,   499,   618,   577,
      15,   620,   371,   619,   619,   499,   619,   476,   617,   239,
     534,   498,   428,   428,   371,   499,   541,    21,   499,   517,
       8,    21,    16,    15,    15,    15,   498,   738,   739,   494,
     502,   770,     7,   499,     7,    21,    21,   371,   613,   503,
     502,   191,   502,   617,   664,   499,   467,   550,     8,    47,
     177,   371,   465,   378,   634,   636,   606,     7,     7,   499,
     723,   724,   721,   722,   461,   499,     5,   620,   766,   767,
     773,   499,   630,     8,    21,    15,    21,    71,   208,   378,
     378,   494,   172,   371,   474,   475,   503,   191,   208,   282,
     285,   290,   298,   790,   791,   792,   799,   811,   812,   813,
     617,   266,   820,   821,   822,   617,    37,   502,   843,   844,
      84,   265,   289,   299,   304,   788,   790,   791,   792,   793,
     794,   795,   797,   798,   799,   617,   790,   791,   792,   793,
     794,   795,   797,   798,   799,   812,   813,   827,   617,   790,
     791,   792,   799,   805,   617,   790,   791,   817,   617,   858,
     858,   858,   378,   859,   860,   858,   858,   503,   763,   330,
     315,   331,   577,   495,   506,    15,    15,    15,    15,    15,
     876,    15,    15,    15,   885,    15,    15,    15,    15,   890,
     351,   352,    15,    15,    15,    15,    15,   896,   371,    18,
      26,   413,    15,   392,     7,   378,   408,   557,   557,   412,
       5,   499,   450,   451,   452,   455,   451,   453,   451,   453,
     246,   246,   246,   246,   246,     8,    37,   371,   438,   502,
       5,   440,   441,     8,    15,    16,    17,   149,   371,   438,
     442,   443,   456,   457,   458,   459,    15,   441,    15,    21,
     520,    21,    21,   509,   577,   499,   510,   553,   567,   579,
     543,   544,   500,   544,   544,   544,   476,   371,   638,   641,
     577,     8,    21,     7,   412,   499,   577,   499,   577,   568,
     631,   499,   621,   622,    21,    21,    21,    21,     8,     8,
     254,   528,   534,    21,   490,   491,   678,   678,   678,    21,
      21,   371,    15,    21,   499,     7,     7,   499,   476,    15,
     173,     8,   668,   669,   670,   671,   672,   674,   675,   676,
     679,   681,   682,   683,   697,   705,   543,   463,    15,    15,
     464,   255,     8,     7,     8,    21,    21,    21,     8,    21,
      21,   708,   709,    15,    15,   371,   371,   472,   473,   475,
      18,     8,    26,   789,    15,   789,   789,    15,   617,   811,
     789,   617,   820,   371,     8,    21,    15,   789,    15,   789,
      15,   617,   788,   617,   827,   617,   805,   617,   817,    21,
      21,    21,   316,   317,     8,    21,    21,    21,    15,    15,
     495,    21,   506,   882,   883,   678,   378,   701,   513,   678,
     707,   721,   707,   663,   663,   506,   893,   499,   891,    15,
      15,   378,   897,   898,   663,   663,   499,   378,   899,    21,
     499,   499,   648,   649,    21,   391,   413,     5,   499,   403,
       8,    21,     8,   516,   516,   516,   516,   516,   447,     5,
      15,   437,   448,   441,   371,   438,   446,   456,   457,   457,
       8,    21,     7,    16,    17,     5,    37,     9,   456,   499,
      20,   509,   496,    21,    26,    21,    21,    21,    21,    15,
     506,   568,   482,   659,   494,   521,   568,   750,   499,    21,
       7,     8,    21,   499,   378,    15,    21,    21,    21,     7,
     771,   772,   773,   499,   499,     7,   678,   502,   665,   378,
     670,    26,   465,    26,   384,   638,   636,   371,   609,   610,
     611,   612,   724,   767,   617,    78,   594,   371,   673,   721,
     698,     8,   371,   475,   499,   617,   800,   378,   617,   617,
     845,   502,   843,   378,   499,   499,   617,   617,   617,   617,
     860,   678,   502,    21,    26,     8,    21,    21,    22,    24,
      33,    35,   158,   159,   161,   162,   192,   234,   247,   702,
     703,   704,     8,    21,    21,    21,    21,    21,    21,    21,
      21,     8,    21,     8,    21,   378,   870,   871,   870,   315,
     350,     8,    21,    21,    21,    21,   348,   349,     8,     8,
      21,     7,    21,    21,   577,   455,   448,   577,   438,    26,
      21,   456,   443,   457,   457,   458,   458,   458,    21,   499,
       5,   499,   517,   639,   640,   502,     8,   678,   502,     8,
     499,   622,   378,    21,   254,   499,     8,    21,   499,    21,
      15,    41,   135,   209,   221,   223,   224,   226,   229,   320,
     322,   499,   464,    21,    21,    15,     8,   132,   768,    21,
      21,     7,    21,   700,   702,   473,     5,    16,    17,    22,
      24,    33,    35,    37,   159,   162,   247,   305,   306,   307,
     802,    21,    94,   230,   284,   295,   814,    37,   191,   288,
     299,   796,    21,    21,    21,    21,   499,   883,    15,    15,
     378,   506,   499,   353,     8,    21,    21,   898,   499,     7,
       7,   411,    21,   495,   442,   456,    21,     8,     8,    21,
     482,   568,   255,    15,    21,   772,     5,   499,   666,   667,
      15,   684,    15,    15,    15,    15,   706,   706,    15,    15,
      15,     8,   498,   610,   710,   711,    15,   721,   699,   699,
       7,     8,    21,   846,    21,     8,   503,   678,   702,     8,
     871,     8,   872,     8,   873,    21,     7,   412,    21,    21,
     499,   640,   499,   371,   623,   624,   499,     8,    21,   685,
     684,   720,   738,   720,   721,   710,   707,   499,   499,   677,
     665,   680,   499,    21,     8,   378,    21,     7,     8,    21,
     678,   801,   499,   378,    21,     8,   499,   378,   378,   371,
     650,   651,    21,     8,    15,    21,   667,   148,   180,   686,
       7,    21,     7,    21,    15,    21,    21,     8,    21,     8,
      21,     8,   710,    78,   701,   701,    21,   329,   499,   503,
     352,   351,     8,   499,    40,   499,   625,   626,   773,     7,
       7,   687,   688,   710,   738,   721,   594,   499,   665,   499,
      21,    21,    21,    15,    21,    15,    15,   651,   371,   627,
       8,    21,     8,    21,    15,    21,    21,    21,     8,   498,
     870,   870,    17,   628,   629,   626,   688,   499,   689,   690,
      21,   499,    21,    21,    21,   630,    17,     7,     8,    21,
       8,   775,   630,   499,   690,    15,   378,   378,   691,   692,
     693,   694,   695,   182,   318,   128,   157,   217,     8,    21,
       7,     7,    15,   696,   696,   696,   692,   378,   694,   695,
     378,   695,   497,     7,    21,   695
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 793 "gram1.y"
    { (yyval.bf_node) = BFNULL; ;}
    break;

  case 3:
#line 795 "gram1.y"
    { (yyval.bf_node) = set_stat_list((yyvsp[(1) - (3)].bf_node),(yyvsp[(2) - (3)].bf_node)); ;}
    break;

  case 4:
#line 799 "gram1.y"
    { lastwasbranch = NO;  (yyval.bf_node) = BFNULL; ;}
    break;

  case 5:
#line 801 "gram1.y"
    {
	       if ((yyvsp[(2) - (3)].bf_node) != BFNULL) 
               {	    
	          (yyvsp[(2) - (3)].bf_node)->label = (yyvsp[(1) - (3)].label);
	          (yyval.bf_node) = (yyvsp[(2) - (3)].bf_node);
	 	  if (is_openmp_stmt) {            /*OMP*/
			is_openmp_stmt = 0;
			if((yyvsp[(2) - (3)].bf_node)) {                        /*OMP*/
				if ((yyvsp[(2) - (3)].bf_node)->decl_specs != -BIT_OPENMP) (yyvsp[(2) - (3)].bf_node)->decl_specs = BIT_OPENMP; /*OMP*/
			}                               /*OMP*/
		  }                                       /*OMP*/
               }
	    ;}
    break;

  case 6:
#line 815 "gram1.y"
    { PTR_BFND p;

	     if(lastwasbranch && ! thislabel)
               /*if (warn_all)
		 warn("statement cannot be reached", 36);*/
	     lastwasbranch = thiswasbranch;
	     thiswasbranch = NO;
	     if((yyvsp[(2) - (3)].bf_node)) (yyvsp[(2) - (3)].bf_node)->label = (yyvsp[(1) - (3)].label);
	     if((yyvsp[(1) - (3)].label) && (yyvsp[(2) - (3)].bf_node)) (yyvsp[(1) - (3)].label)->statbody = (yyvsp[(2) - (3)].bf_node); /*8.11.06 podd*/
	     if((yyvsp[(1) - (3)].label)) {
		/*$1->statbody = $2;*/ /*8.11.06 podd*/
		if((yyvsp[(1) - (3)].label)->labtype == LABFORMAT)
		  err("label already that of a format",39);
		else
		  (yyvsp[(1) - (3)].label)->labtype = LABEXEC;
	     }
	     if (is_openmp_stmt) {            /*OMP*/
			is_openmp_stmt = 0;
			if((yyvsp[(2) - (3)].bf_node)) {                        /*OMP*/
				if ((yyvsp[(2) - (3)].bf_node)->decl_specs != -BIT_OPENMP) (yyvsp[(2) - (3)].bf_node)->decl_specs = BIT_OPENMP; /*OMP*/
			}                               /*OMP*/
	     }                                       /*OMP*/
             for (p = pred_bfnd; (yyvsp[(1) - (3)].label) && 
		  ((p->variant == FOR_NODE)||(p->variant == WHILE_NODE)) &&
                  (p->entry.for_node.doend) &&
		  (p->entry.for_node.doend->stateno == (yyvsp[(1) - (3)].label)->stateno);
		  p = p->control_parent)
                ++end_group;
	     (yyval.bf_node) = (yyvsp[(2) - (3)].bf_node);
     ;}
    break;

  case 7:
#line 846 "gram1.y"
    { /* PTR_LLND p; */
			doinclude( (yyvsp[(3) - (3)].charp) );
/*			p = make_llnd(fi, STRING_VAL, LLNULL, LLNULL, SMNULL);
			p->entry.string_val = $3;
			p->type = global_string;
			$$ = get_bfnd(fi, INCLUDE_STAT, SMNULL, p, LLNULL); */
			(yyval.bf_node) = BFNULL;
		;}
    break;

  case 8:
#line 855 "gram1.y"
    {
	      err("Unclassifiable statement", 10);
	      flline();
	      (yyval.bf_node) = BFNULL;
	    ;}
    break;

  case 9:
#line 861 "gram1.y"
    { PTR_CMNT p;
              PTR_BFND bif; 
	    
              if (last_bfnd && last_bfnd->control_parent &&((last_bfnd->control_parent->variant == LOGIF_NODE)
	         ||(last_bfnd->control_parent->variant == FORALL_STAT)))
  	         bif = last_bfnd->control_parent;
              else
                 bif = last_bfnd;
              p=bif->entry.Template.cmnt_ptr;
              if(p)
                 p->string = StringConcatenation(p->string,commentbuf);
              else
              {
                 p = make_comment(fi,commentbuf, FULL);
                 bif->entry.Template.cmnt_ptr = p;
              }
 	      (yyval.bf_node) = BFNULL;         
            ;}
    break;

  case 10:
#line 881 "gram1.y"
    { 
	      flline();	 needkwd = NO;	inioctl = NO;
/*!!!*/
              opt_kwd_ = NO; intonly = NO; opt_kwd_hedr = NO; opt_kwd_r = NO; as_op_kwd_= NO; optcorner = NO;
	      yyerrok; yyclearin;  (yyval.bf_node) = BFNULL;
	    ;}
    break;

  case 11:
#line 890 "gram1.y"
    {
	    if(yystno)
	      {
	      (yyval.label) = thislabel =	make_label_node(fi,yystno);
	      thislabel->scope = cur_scope();
	      if (thislabel->labdefined && (thislabel->scope == cur_scope()))
		 errstr("Label %s already defined",convic(thislabel->stateno),40);
	      else
		 thislabel->labdefined = YES;
	      }
	    else
	      (yyval.label) = thislabel = LBNULL;
	    ;}
    break;

  case 12:
#line 906 "gram1.y"
    { PTR_BFND p;

	        if (pred_bfnd != global_bfnd)
		    err("Misplaced PROGRAM statement", 33);
		p = get_bfnd(fi,PROG_HEDR, (yyvsp[(3) - (3)].symbol), LLNULL, LLNULL, LLNULL);
		(yyvsp[(3) - (3)].symbol)->entry.prog_decl.prog_hedr=p;
 		set_blobs(p, global_bfnd, NEW_GROUP1);
	        add_scope_level(p, NO);
	        position = IN_PROC;
	    ;}
    break;

  case 13:
#line 918 "gram1.y"
    {  PTR_BFND q = BFNULL;

	      (yyvsp[(3) - (3)].symbol)->variant = PROCEDURE_NAME;
	      (yyvsp[(3) - (3)].symbol)->decl = YES;   /* variable declaration has been seen. */
	      q = get_bfnd(fi,BLOCK_DATA, (yyvsp[(3) - (3)].symbol), LLNULL, LLNULL, LLNULL);
	      set_blobs(q, global_bfnd, NEW_GROUP1);
              add_scope_level(q, NO);
	    ;}
    break;

  case 14:
#line 928 "gram1.y"
    { 
              install_param_list((yyvsp[(3) - (4)].symbol), (yyvsp[(4) - (4)].symbol), LLNULL, PROCEDURE_NAME); 
	      /* if there is only a control end the control parent is not set */
              
	     ;}
    break;

  case 15:
#line 935 "gram1.y"
    { install_param_list((yyvsp[(4) - (5)].symbol), (yyvsp[(5) - (5)].symbol), LLNULL, PROCEDURE_NAME); 
              if((yyvsp[(1) - (5)].ll_node)->variant == RECURSIVE_OP) 
                   (yyvsp[(4) - (5)].symbol)->attr = (yyvsp[(4) - (5)].symbol)->attr | RECURSIVE_BIT;
              pred_bfnd->entry.Template.ll_ptr3 = (yyvsp[(1) - (5)].ll_node);
            ;}
    break;

  case 16:
#line 941 "gram1.y"
    {
              install_param_list((yyvsp[(3) - (5)].symbol), (yyvsp[(4) - (5)].symbol), (yyvsp[(5) - (5)].ll_node), FUNCTION_NAME);  
  	      pred_bfnd->entry.Template.ll_ptr1 = (yyvsp[(5) - (5)].ll_node);
            ;}
    break;

  case 17:
#line 946 "gram1.y"
    {
              install_param_list((yyvsp[(1) - (3)].symbol), (yyvsp[(2) - (3)].symbol), (yyvsp[(3) - (3)].ll_node), FUNCTION_NAME); 
	      pred_bfnd->entry.Template.ll_ptr1 = (yyvsp[(3) - (3)].ll_node);
	    ;}
    break;

  case 18:
#line 951 "gram1.y"
    {PTR_BFND p, bif;
	     PTR_SYMB q = SMNULL;
             PTR_LLND l = LLNULL;

	     if(parstate==OUTSIDE || procclass==CLMAIN || procclass==CLBLOCK)
	        err("Misplaced ENTRY statement", 35);

	     bif = cur_scope();
	     if (bif->variant == FUNC_HEDR) {
	        q = make_function((yyvsp[(2) - (4)].hash_entry), bif->entry.Template.symbol->type, LOCAL);
	        l = construct_entry_list(q, (yyvsp[(3) - (4)].symbol), FUNCTION_NAME); 
             }
             else if ((bif->variant == PROC_HEDR) || 
                      (bif->variant == PROS_HEDR) || /* added for FORTRAN M */
                      (bif->variant == PROG_HEDR)) {
	             q = make_procedure((yyvsp[(2) - (4)].hash_entry),LOCAL);
  	             l = construct_entry_list(q, (yyvsp[(3) - (4)].symbol), PROCEDURE_NAME); 
             }
	     p = get_bfnd(fi,ENTRY_STAT, q, l, (yyvsp[(4) - (4)].ll_node), LLNULL);
	     set_blobs(p, pred_bfnd, SAME_GROUP);
             q->decl = YES;   /*4.02.03*/
             q->entry.proc_decl.proc_hedr = p; /*5.02.03*/
	    ;}
    break;

  case 19:
#line 975 "gram1.y"
    { PTR_SYMB s;
	      PTR_BFND p;
/*
	      s = make_global_entity($3, MODULE_NAME, global_default, NO);
	      s->decl = YES;  
	      p = get_bfnd(fi, MODULE_STMT, s, LLNULL, LLNULL, LLNULL);
	      s->entry.Template.func_hedr = p;
	      set_blobs(p, pred_bfnd, NEW_GROUP1);
              add_scope_level(p, NO);
*/
	      /*position = IN_MODULE;*/


               s = make_module((yyvsp[(3) - (3)].hash_entry));
	       s->decl = YES;   /* variable declaration has been seen. */
	        if (pred_bfnd != global_bfnd)
		    err("Misplaced MODULE statement", 33);
              p = get_bfnd(fi, MODULE_STMT, s, LLNULL, LLNULL, LLNULL);
	      s->entry.Template.func_hedr = p; /* !!!????*/
	      set_blobs(p, global_bfnd, NEW_GROUP1);
	      add_scope_level(p, NO);	
	      position =  IN_MODULE;    /*IN_PROC*/
              privateall = 0;
            ;}
    break;

  case 20:
#line 1001 "gram1.y"
    { newprog(); 
	      if (position == IN_OUTSIDE)
	           position = IN_PROC;
              else if (position != IN_INTERNAL_PROC){ 
                if(!is_interface_stat(pred_bfnd))
	           position--;
              }
              else {
                if(!is_interface_stat(pred_bfnd))
                  err("Internal procedures can not contain procedures",304);
              }
	    ;}
    break;

  case 21:
#line 1016 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, RECURSIVE_OP, LLNULL, LLNULL, SMNULL); ;}
    break;

  case 22:
#line 1018 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, PURE_OP, LLNULL, LLNULL, SMNULL); ;}
    break;

  case 23:
#line 1020 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, ELEMENTAL_OP, LLNULL, LLNULL, SMNULL); ;}
    break;

  case 24:
#line 1024 "gram1.y"
    { PTR_BFND p;

	      (yyval.symbol) = make_procedure((yyvsp[(1) - (1)].hash_entry), LOCAL);
	      (yyval.symbol)->decl = YES;   /* variable declaration has been seen. */
             /* if (pred_bfnd != global_bfnd)
		 {
	         err("Misplaced SUBROUTINE statement", 34);
		 }  
              */
	      p = get_bfnd(fi,PROC_HEDR, (yyval.symbol), LLNULL, LLNULL, LLNULL);
              (yyval.symbol)->entry.proc_decl.proc_hedr = p;
	      set_blobs(p, pred_bfnd, NEW_GROUP1);
              add_scope_level(p, NO);
            ;}
    break;

  case 25:
#line 1041 "gram1.y"
    { PTR_BFND p;

	      (yyval.symbol) = make_function((yyvsp[(1) - (1)].hash_entry), TYNULL, LOCAL);
	      (yyval.symbol)->decl = YES;   /* variable declaration has been seen. */
             /* if (pred_bfnd != global_bfnd)
	         err("Misplaced FUNCTION statement", 34); */
	      p = get_bfnd(fi,FUNC_HEDR, (yyval.symbol), LLNULL, LLNULL, LLNULL);
              (yyval.symbol)->entry.func_decl.func_hedr = p;
	      set_blobs(p, pred_bfnd, NEW_GROUP1);
              add_scope_level(p, NO);
            ;}
    break;

  case 26:
#line 1055 "gram1.y"
    { PTR_BFND p;
             PTR_LLND l;

	      (yyval.symbol) = make_function((yyvsp[(4) - (4)].hash_entry), (yyvsp[(1) - (4)].data_type), LOCAL);
	      (yyval.symbol)->decl = YES;   /* variable declaration has been seen. */
              l = make_llnd(fi, TYPE_OP, LLNULL, LLNULL, SMNULL);
              l->type = (yyvsp[(1) - (4)].data_type);
	      p = get_bfnd(fi,FUNC_HEDR, (yyval.symbol), LLNULL, l, LLNULL);
              (yyval.symbol)->entry.func_decl.func_hedr = p;
            /*  if (pred_bfnd != global_bfnd)
	         err("Misplaced FUNCTION statement", 34);*/
	      set_blobs(p, pred_bfnd, NEW_GROUP1);
              add_scope_level(p, NO);
/*
	      $$ = make_function($4, $1, LOCAL);
	      $$->decl = YES;
	      p = get_bfnd(fi,FUNC_HEDR, $$, LLNULL, LLNULL, LLNULL);
              if (pred_bfnd != global_bfnd)
	         errstr("cftn.gram: misplaced SUBROUTINE statement.");
	      set_blobs(p, pred_bfnd, NEW_GROUP1);
              add_scope_level(p, NO);
*/
           ;}
    break;

  case 27:
#line 1079 "gram1.y"
    { PTR_BFND p;
             PTR_LLND l;
	      (yyval.symbol) = make_function((yyvsp[(5) - (5)].hash_entry), (yyvsp[(1) - (5)].data_type), LOCAL);
	      (yyval.symbol)->decl = YES;   /* variable declaration has been seen. */
              if((yyvsp[(2) - (5)].ll_node)->variant == RECURSIVE_OP)
	         (yyval.symbol)->attr = (yyval.symbol)->attr | RECURSIVE_BIT;
              l = make_llnd(fi, TYPE_OP, LLNULL, LLNULL, SMNULL);
              l->type = (yyvsp[(1) - (5)].data_type);
             /* if (pred_bfnd != global_bfnd)
	         err("Misplaced FUNCTION statement", 34);*/
	      p = get_bfnd(fi,FUNC_HEDR, (yyval.symbol), LLNULL, l, (yyvsp[(2) - (5)].ll_node));
              (yyval.symbol)->entry.func_decl.func_hedr = p;
	      set_blobs(p, pred_bfnd, NEW_GROUP1);
              add_scope_level(p, NO);
            ;}
    break;

  case 28:
#line 1095 "gram1.y"
    { PTR_BFND p;

	      (yyval.symbol) = make_function((yyvsp[(4) - (4)].hash_entry), TYNULL, LOCAL);
	      (yyval.symbol)->decl = YES;   /* variable declaration has been seen. */
              if((yyvsp[(1) - (4)].ll_node)->variant == RECURSIVE_OP)
	        (yyval.symbol)->attr = (yyval.symbol)->attr | RECURSIVE_BIT;
              /*if (pred_bfnd != global_bfnd)
	         err("Misplaced FUNCTION statement",34);*/
	      p = get_bfnd(fi,FUNC_HEDR, (yyval.symbol), LLNULL, LLNULL, (yyvsp[(1) - (4)].ll_node));
              (yyval.symbol)->entry.func_decl.func_hedr = p;
	      set_blobs(p, pred_bfnd, NEW_GROUP1);
              add_scope_level(p, NO);
            ;}
    break;

  case 29:
#line 1109 "gram1.y"
    { PTR_BFND p;
              PTR_LLND l;
	      (yyval.symbol) = make_function((yyvsp[(5) - (5)].hash_entry), (yyvsp[(2) - (5)].data_type), LOCAL);
	      (yyval.symbol)->decl = YES;   /* variable declaration has been seen. */
              if((yyvsp[(1) - (5)].ll_node)->variant == RECURSIVE_OP)
	        (yyval.symbol)->attr = (yyval.symbol)->attr | RECURSIVE_BIT;
              l = make_llnd(fi, TYPE_OP, LLNULL, LLNULL, SMNULL);
              l->type = (yyvsp[(2) - (5)].data_type);
             /* if (pred_bfnd != global_bfnd)
	          err("Misplaced FUNCTION statement",34);*/
	      p = get_bfnd(fi,FUNC_HEDR, (yyval.symbol), LLNULL, l, (yyvsp[(1) - (5)].ll_node));
              (yyval.symbol)->entry.func_decl.func_hedr = p;
	      set_blobs(p, pred_bfnd, NEW_GROUP1);
              add_scope_level(p, NO);
            ;}
    break;

  case 30:
#line 1127 "gram1.y"
    { (yyval.ll_node) = LLNULL; ;}
    break;

  case 31:
#line 1129 "gram1.y"
    { PTR_SYMB s;
              s = make_scalar((yyvsp[(4) - (5)].hash_entry), TYNULL, LOCAL);
              (yyval.ll_node) = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
            ;}
    break;

  case 32:
#line 1136 "gram1.y"
    { (yyval.hash_entry) = look_up_sym(yytext); ;}
    break;

  case 33:
#line 1139 "gram1.y"
    { (yyval.symbol) = make_program(look_up_sym("_MAIN")); ;}
    break;

  case 34:
#line 1141 "gram1.y"
    {
              (yyval.symbol) = make_program((yyvsp[(1) - (1)].hash_entry));
	      (yyval.symbol)->decl = YES;   /* variable declaration has been seen. */
            ;}
    break;

  case 35:
#line 1147 "gram1.y"
    { (yyval.symbol) = make_program(look_up_sym("_BLOCK")); ;}
    break;

  case 36:
#line 1149 "gram1.y"
    {
              (yyval.symbol) = make_program((yyvsp[(1) - (1)].hash_entry)); 
	      (yyval.symbol)->decl = YES;   /* variable declaration has been seen. */
	    ;}
    break;

  case 37:
#line 1156 "gram1.y"
    { (yyval.symbol) = SMNULL; ;}
    break;

  case 38:
#line 1158 "gram1.y"
    { (yyval.symbol) = SMNULL; ;}
    break;

  case 39:
#line 1160 "gram1.y"
    { (yyval.symbol) = (yyvsp[(2) - (3)].symbol); ;}
    break;

  case 41:
#line 1165 "gram1.y"
    { (yyval.symbol) = set_id_list((yyvsp[(1) - (3)].symbol), (yyvsp[(3) - (3)].symbol)); ;}
    break;

  case 42:
#line 1169 "gram1.y"
    {
	      (yyval.symbol) = make_scalar((yyvsp[(1) - (1)].hash_entry), TYNULL, IO);
            ;}
    break;

  case 43:
#line 1173 "gram1.y"
    { (yyval.symbol) = make_scalar(look_up_sym("*"), TYNULL, IO); ;}
    break;

  case 44:
#line 1179 "gram1.y"
    { char *s;

	      s = copyn(yyleng+1, yytext);
	      s[yyleng] = '\0';
	      (yyval.charp) = s;
	    ;}
    break;

  case 45:
#line 1188 "gram1.y"
    { needkwd = 1; ;}
    break;

  case 46:
#line 1192 "gram1.y"
    { needkwd = NO; ;}
    break;

  case 47:
#line 1197 "gram1.y"
    { colon_flag = YES; ;}
    break;

  case 61:
#line 1218 "gram1.y"
    {
	      saveall = YES;
	      (yyval.bf_node) = get_bfnd(fi,SAVE_DECL, SMNULL, LLNULL, LLNULL, LLNULL);
	    ;}
    break;

  case 62:
#line 1223 "gram1.y"
    {
	      (yyval.bf_node) = get_bfnd(fi,SAVE_DECL, SMNULL, (yyvsp[(4) - (4)].ll_node), LLNULL, LLNULL);
            ;}
    break;

  case 63:
#line 1228 "gram1.y"
    { PTR_LLND p;

	      p = make_llnd(fi,STMT_STR, LLNULL, LLNULL, SMNULL);
	      p->entry.string_val = copys(stmtbuf);
	      (yyval.bf_node) = get_bfnd(fi,FORMAT_STAT, SMNULL, p, LLNULL, LLNULL);
             ;}
    break;

  case 64:
#line 1235 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,PARAM_DECL, SMNULL, (yyvsp[(4) - (5)].ll_node), LLNULL, LLNULL); ;}
    break;

  case 77:
#line 1251 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, INTERFACE_STMT, SMNULL, LLNULL, LLNULL, LLNULL); 
              add_scope_level((yyval.bf_node), NO);     
            ;}
    break;

  case 78:
#line 1255 "gram1.y"
    { PTR_SYMB s;

	      s = make_procedure((yyvsp[(3) - (3)].hash_entry), LOCAL);
	      s->variant = INTERFACE_NAME;
	      (yyval.bf_node) = get_bfnd(fi, INTERFACE_STMT, s, LLNULL, LLNULL, LLNULL);
              add_scope_level((yyval.bf_node), NO);
	    ;}
    break;

  case 79:
#line 1263 "gram1.y"
    { PTR_SYMB s;

	      s = make_function((yyvsp[(4) - (5)].hash_entry), global_default, LOCAL);
	      s->variant = INTERFACE_NAME;
	      (yyval.bf_node) = get_bfnd(fi, INTERFACE_OPERATOR, s, LLNULL, LLNULL, LLNULL);
              add_scope_level((yyval.bf_node), NO);
	    ;}
    break;

  case 80:
#line 1271 "gram1.y"
    { PTR_SYMB s;


	      s = make_procedure(look_up_sym("="), LOCAL);
	      s->variant = INTERFACE_NAME;
	      (yyval.bf_node) = get_bfnd(fi, INTERFACE_ASSIGNMENT, s, LLNULL, LLNULL, LLNULL);
              add_scope_level((yyval.bf_node), NO);
	    ;}
    break;

  case 81:
#line 1280 "gram1.y"
    { parstate = INDCL;
              (yyval.bf_node) = get_bfnd(fi, CONTROL_END, SMNULL, LLNULL, LLNULL, LLNULL); 
	      /*process_interface($$);*/ /*podd 01.02.03*/
              delete_beyond_scope_level(pred_bfnd);
	    ;}
    break;

  case 82:
#line 1288 "gram1.y"
    { (yyval.hash_entry) = look_up_sym(yytext); ;}
    break;

  case 83:
#line 1292 "gram1.y"
    { (yyval.hash_entry) = (yyvsp[(1) - (1)].hash_entry); ;}
    break;

  case 84:
#line 1294 "gram1.y"
    { (yyval.hash_entry) = (yyvsp[(1) - (1)].hash_entry); ;}
    break;

  case 85:
#line 1298 "gram1.y"
    { (yyval.hash_entry) = look_up_op(PLUS); ;}
    break;

  case 86:
#line 1300 "gram1.y"
    { (yyval.hash_entry) = look_up_op(MINUS); ;}
    break;

  case 87:
#line 1302 "gram1.y"
    { (yyval.hash_entry) = look_up_op(ASTER); ;}
    break;

  case 88:
#line 1304 "gram1.y"
    { (yyval.hash_entry) = look_up_op(DASTER); ;}
    break;

  case 89:
#line 1306 "gram1.y"
    { (yyval.hash_entry) = look_up_op(SLASH); ;}
    break;

  case 90:
#line 1308 "gram1.y"
    { (yyval.hash_entry) = look_up_op(DSLASH); ;}
    break;

  case 91:
#line 1310 "gram1.y"
    { (yyval.hash_entry) = look_up_op(AND); ;}
    break;

  case 92:
#line 1312 "gram1.y"
    { (yyval.hash_entry) = look_up_op(OR); ;}
    break;

  case 93:
#line 1314 "gram1.y"
    { (yyval.hash_entry) = look_up_op(XOR); ;}
    break;

  case 94:
#line 1316 "gram1.y"
    { (yyval.hash_entry) = look_up_op(NOT); ;}
    break;

  case 95:
#line 1318 "gram1.y"
    { (yyval.hash_entry) = look_up_op(EQ); ;}
    break;

  case 96:
#line 1320 "gram1.y"
    { (yyval.hash_entry) = look_up_op(NE); ;}
    break;

  case 97:
#line 1322 "gram1.y"
    { (yyval.hash_entry) = look_up_op(GT); ;}
    break;

  case 98:
#line 1324 "gram1.y"
    { (yyval.hash_entry) = look_up_op(GE); ;}
    break;

  case 99:
#line 1326 "gram1.y"
    { (yyval.hash_entry) = look_up_op(LT); ;}
    break;

  case 100:
#line 1328 "gram1.y"
    { (yyval.hash_entry) = look_up_op(LE); ;}
    break;

  case 101:
#line 1330 "gram1.y"
    { (yyval.hash_entry) = look_up_op(NEQV); ;}
    break;

  case 102:
#line 1332 "gram1.y"
    { (yyval.hash_entry) = look_up_op(EQV); ;}
    break;

  case 103:
#line 1337 "gram1.y"
    {
             PTR_SYMB s;
         
             type_var = s = make_derived_type((yyvsp[(4) - (4)].hash_entry), TYNULL, LOCAL);	
             (yyval.bf_node) = get_bfnd(fi, STRUCT_DECL, s, LLNULL, LLNULL, LLNULL);
             add_scope_level((yyval.bf_node), NO);
	   ;}
    break;

  case 104:
#line 1346 "gram1.y"
    { PTR_SYMB s;
         
             type_var = s = make_derived_type((yyvsp[(7) - (7)].hash_entry), TYNULL, LOCAL);	
	     s->attr = s->attr | type_opt;
             (yyval.bf_node) = get_bfnd(fi, STRUCT_DECL, s, (yyvsp[(5) - (7)].ll_node), LLNULL, LLNULL);
             add_scope_level((yyval.bf_node), NO);
	   ;}
    break;

  case 105:
#line 1356 "gram1.y"
    {
	     (yyval.bf_node) = get_bfnd(fi, CONTROL_END, SMNULL, LLNULL, LLNULL, LLNULL);
	     if (type_var != SMNULL)
               process_type(type_var, (yyval.bf_node));
             type_var = SMNULL;
	     delete_beyond_scope_level(pred_bfnd);
           ;}
    break;

  case 106:
#line 1364 "gram1.y"
    {
             (yyval.bf_node) = get_bfnd(fi, CONTROL_END, SMNULL, LLNULL, LLNULL, LLNULL);
	     if (type_var != SMNULL)
               process_type(type_var, (yyval.bf_node));
             type_var = SMNULL;
	     delete_beyond_scope_level(pred_bfnd);	
           ;}
    break;

  case 107:
#line 1374 "gram1.y"
    { 
	      PTR_LLND q, r, l;
	     /* PTR_SYMB s;*/
	      PTR_TYPE t;
	      int type_opts;

	      vartype = (yyvsp[(1) - (7)].data_type);
              if((yyvsp[(6) - (7)].ll_node) && vartype->variant != T_STRING)
                errstr("Non character entity  %s  has length specification",(yyvsp[(3) - (7)].hash_entry)->ident,41);
              t = make_type_node(vartype, (yyvsp[(6) - (7)].ll_node));
	      type_opts = type_options;
	      if ((yyvsp[(5) - (7)].ll_node)) type_opts = type_opts | DIMENSION_BIT;
	      if ((yyvsp[(5) - (7)].ll_node))
		 q = deal_with_options((yyvsp[(3) - (7)].hash_entry), t, type_opts, (yyvsp[(5) - (7)].ll_node), ndim, (yyvsp[(7) - (7)].ll_node), (yyvsp[(5) - (7)].ll_node));
	      else q = deal_with_options((yyvsp[(3) - (7)].hash_entry), t, type_opts, attr_dims, attr_ndim, (yyvsp[(7) - (7)].ll_node), (yyvsp[(5) - (7)].ll_node));
	      r = make_llnd(fi, EXPR_LIST, q, LLNULL, SMNULL);
	      l = make_llnd(fi, TYPE_OP, LLNULL, LLNULL, SMNULL);
	      l->type = vartype;
	      (yyval.bf_node) = get_bfnd(fi,VAR_DECL, SMNULL, r, l, (yyvsp[(2) - (7)].ll_node));
	    ;}
    break;

  case 108:
#line 1395 "gram1.y"
    { 
	      PTR_LLND q, r;
	    /*  PTR_SYMB s;*/
              PTR_TYPE t;
	      int type_opts;
              if((yyvsp[(5) - (6)].ll_node) && vartype->variant != T_STRING)
                errstr("Non character entity  %s  has length specification",(yyvsp[(3) - (6)].hash_entry)->ident,41);
              t = make_type_node(vartype, (yyvsp[(5) - (6)].ll_node));
	      type_opts = type_options;
	      if ((yyvsp[(4) - (6)].ll_node)) type_opts = type_opts | DIMENSION_BIT;
	      if ((yyvsp[(4) - (6)].ll_node))
		 q = deal_with_options((yyvsp[(3) - (6)].hash_entry), t, type_opts, (yyvsp[(4) - (6)].ll_node), ndim, (yyvsp[(6) - (6)].ll_node), (yyvsp[(4) - (6)].ll_node));
	      else q = deal_with_options((yyvsp[(3) - (6)].hash_entry), t, type_opts, attr_dims, attr_ndim, (yyvsp[(6) - (6)].ll_node), (yyvsp[(4) - (6)].ll_node));
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      add_to_lowLevelList(r, (yyvsp[(1) - (6)].bf_node)->entry.Template.ll_ptr1);
       	    ;}
    break;

  case 109:
#line 1414 "gram1.y"
    { (yyval.ll_node) = LLNULL; ;}
    break;

  case 110:
#line 1416 "gram1.y"
    { (yyval.ll_node) = LLNULL; ;}
    break;

  case 111:
#line 1418 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(3) - (5)].ll_node); ;}
    break;

  case 112:
#line 1422 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 113:
#line 1424 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (4)].ll_node), (yyvsp[(4) - (4)].ll_node), EXPR_LIST); ;}
    break;

  case 114:
#line 1428 "gram1.y"
    { type_options = type_options | PARAMETER_BIT; 
              (yyval.ll_node) = make_llnd(fi, PARAMETER_OP, LLNULL, LLNULL, SMNULL);
            ;}
    break;

  case 115:
#line 1432 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 116:
#line 1434 "gram1.y"
    { type_options = type_options | ALLOCATABLE_BIT;
              (yyval.ll_node) = make_llnd(fi, ALLOCATABLE_OP, LLNULL, LLNULL, SMNULL);
	    ;}
    break;

  case 117:
#line 1438 "gram1.y"
    { type_options = type_options | DIMENSION_BIT;
	      attr_ndim = ndim;
	      attr_dims = (yyvsp[(2) - (2)].ll_node);
              (yyval.ll_node) = make_llnd(fi, DIMENSION_OP, (yyvsp[(2) - (2)].ll_node), LLNULL, SMNULL);
            ;}
    break;

  case 118:
#line 1444 "gram1.y"
    { type_options = type_options | EXTERNAL_BIT;
              (yyval.ll_node) = make_llnd(fi, EXTERNAL_OP, LLNULL, LLNULL, SMNULL);
            ;}
    break;

  case 119:
#line 1448 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(3) - (4)].ll_node); ;}
    break;

  case 120:
#line 1450 "gram1.y"
    { type_options = type_options | INTRINSIC_BIT;
              (yyval.ll_node) = make_llnd(fi, INTRINSIC_OP, LLNULL, LLNULL, SMNULL);
            ;}
    break;

  case 121:
#line 1454 "gram1.y"
    { type_options = type_options | OPTIONAL_BIT;
              (yyval.ll_node) = make_llnd(fi, OPTIONAL_OP, LLNULL, LLNULL, SMNULL);
            ;}
    break;

  case 122:
#line 1458 "gram1.y"
    { type_options = type_options | POINTER_BIT;
              (yyval.ll_node) = make_llnd(fi, POINTER_OP, LLNULL, LLNULL, SMNULL);
            ;}
    break;

  case 123:
#line 1462 "gram1.y"
    { type_options = type_options | SAVE_BIT; 
              (yyval.ll_node) = make_llnd(fi, SAVE_OP, LLNULL, LLNULL, SMNULL);
            ;}
    break;

  case 124:
#line 1466 "gram1.y"
    { type_options = type_options | SAVE_BIT; 
              (yyval.ll_node) = make_llnd(fi, STATIC_OP, LLNULL, LLNULL, SMNULL);
            ;}
    break;

  case 125:
#line 1470 "gram1.y"
    { type_options = type_options | TARGET_BIT; 
              (yyval.ll_node) = make_llnd(fi, TARGET_OP, LLNULL, LLNULL, SMNULL);
            ;}
    break;

  case 126:
#line 1476 "gram1.y"
    { type_options = type_options | IN_BIT;  type_opt = IN_BIT; 
              (yyval.ll_node) = make_llnd(fi, IN_OP, LLNULL, LLNULL, SMNULL);
            ;}
    break;

  case 127:
#line 1480 "gram1.y"
    { type_options = type_options | OUT_BIT;  type_opt = OUT_BIT; 
              (yyval.ll_node) = make_llnd(fi, OUT_OP, LLNULL, LLNULL, SMNULL);
            ;}
    break;

  case 128:
#line 1484 "gram1.y"
    { type_options = type_options | INOUT_BIT;  type_opt = INOUT_BIT;
              (yyval.ll_node) = make_llnd(fi, INOUT_OP, LLNULL, LLNULL, SMNULL);
            ;}
    break;

  case 129:
#line 1490 "gram1.y"
    { type_options = type_options | PUBLIC_BIT; 
              type_opt = PUBLIC_BIT;
              (yyval.ll_node) = make_llnd(fi, PUBLIC_OP, LLNULL, LLNULL, SMNULL);
            ;}
    break;

  case 130:
#line 1495 "gram1.y"
    { type_options =  type_options | PRIVATE_BIT;
               type_opt = PRIVATE_BIT;
              (yyval.ll_node) = make_llnd(fi, PRIVATE_OP, LLNULL, LLNULL, SMNULL);
            ;}
    break;

  case 131:
#line 1502 "gram1.y"
    { 
	      PTR_LLND q, r;
	      PTR_SYMB s;

              s = make_scalar((yyvsp[(7) - (7)].hash_entry), TYNULL, LOCAL);
	      s->attr = s->attr | type_opt;	
	      q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      (yyval.bf_node) = get_bfnd(fi, INTENT_STMT, SMNULL, r, (yyvsp[(4) - (7)].ll_node), LLNULL);
	    ;}
    break;

  case 132:
#line 1513 "gram1.y"
    { 
	      PTR_LLND q, r;
	      PTR_SYMB s;

              s = make_scalar((yyvsp[(3) - (3)].hash_entry), TYNULL, LOCAL);	
	      s->attr = s->attr | type_opt;
	      q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      add_to_lowLevelList(r, (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);
  	    ;}
    break;

  case 133:
#line 1526 "gram1.y"
    { 
	      PTR_LLND q, r;
	      PTR_SYMB s;

              s = make_scalar((yyvsp[(4) - (4)].hash_entry), TYNULL, LOCAL);	
	      s->attr = s->attr | OPTIONAL_BIT;
	      q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      (yyval.bf_node) = get_bfnd(fi, OPTIONAL_STMT, SMNULL, r, LLNULL, LLNULL);
	    ;}
    break;

  case 134:
#line 1537 "gram1.y"
    { 
	      PTR_LLND q, r;
	      PTR_SYMB s;

              s = make_scalar((yyvsp[(3) - (3)].hash_entry), TYNULL, LOCAL);	
	      s->attr = s->attr | OPTIONAL_BIT;
	      q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      add_to_lowLevelList(r, (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);
  	    ;}
    break;

  case 135:
#line 1550 "gram1.y"
    { 
	      PTR_LLND r;
	      PTR_SYMB s;

              s = (yyvsp[(4) - (4)].ll_node)->entry.Template.symbol; 
              s->attr = s->attr | SAVE_BIT;
	      r = make_llnd(fi,EXPR_LIST, (yyvsp[(4) - (4)].ll_node), LLNULL, SMNULL);
	      (yyval.bf_node) = get_bfnd(fi, STATIC_STMT, SMNULL, r, LLNULL, LLNULL);
	    ;}
    break;

  case 136:
#line 1560 "gram1.y"
    { 
	      PTR_LLND r;
	      PTR_SYMB s;

              s = (yyvsp[(3) - (3)].ll_node)->entry.Template.symbol;
              s->attr = s->attr | SAVE_BIT;
	      r = make_llnd(fi,EXPR_LIST, (yyvsp[(3) - (3)].ll_node), LLNULL, SMNULL);
	      add_to_lowLevelList(r, (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);
  	    ;}
    break;

  case 137:
#line 1573 "gram1.y"
    {
	      privateall = 1;
	      (yyval.bf_node) = get_bfnd(fi, PRIVATE_STMT, SMNULL, LLNULL, LLNULL, LLNULL);
	    ;}
    break;

  case 138:
#line 1578 "gram1.y"
    {
	      /*type_options = type_options | PRIVATE_BIT;*/
	      (yyval.bf_node) = get_bfnd(fi, PRIVATE_STMT, SMNULL, (yyvsp[(5) - (5)].ll_node), LLNULL, LLNULL);
            ;}
    break;

  case 139:
#line 1584 "gram1.y"
    {type_opt = PRIVATE_BIT;;}
    break;

  case 140:
#line 1588 "gram1.y"
    { 
	      (yyval.bf_node) = get_bfnd(fi, SEQUENCE_STMT, SMNULL, LLNULL, LLNULL, LLNULL);
            ;}
    break;

  case 141:
#line 1593 "gram1.y"
    {
	      /*saveall = YES;*/ /*14.03.03*/
	      (yyval.bf_node) = get_bfnd(fi, PUBLIC_STMT, SMNULL, LLNULL, LLNULL, LLNULL);
	    ;}
    break;

  case 142:
#line 1598 "gram1.y"
    {
	      /*type_options = type_options | PUBLIC_BIT;*/
	      (yyval.bf_node) = get_bfnd(fi, PUBLIC_STMT, SMNULL, (yyvsp[(5) - (5)].ll_node), LLNULL, LLNULL);
            ;}
    break;

  case 143:
#line 1604 "gram1.y"
    {type_opt = PUBLIC_BIT;;}
    break;

  case 144:
#line 1608 "gram1.y"
    {
	      type_options = 0;
              /* following block added by dbg */
	      ndim = 0;
	      attr_ndim = 0;
	      attr_dims = LLNULL;
	      /* end section added by dbg */
              (yyval.data_type) = make_type_node((yyvsp[(1) - (4)].data_type), (yyvsp[(3) - (4)].ll_node));
            ;}
    break;

  case 145:
#line 1618 "gram1.y"
    { PTR_TYPE t;

	      type_options = 0;
	      ndim = 0;
	      attr_ndim = 0;
	      attr_dims = LLNULL;
              t = lookup_type((yyvsp[(3) - (5)].hash_entry));
	      vartype = t;
	      (yyval.data_type) = make_type_node(t, LLNULL);
            ;}
    break;

  case 146:
#line 1631 "gram1.y"
    {opt_kwd_hedr = YES;;}
    break;

  case 147:
#line 1636 "gram1.y"
    { PTR_TYPE p;
	      PTR_LLND q;
	      PTR_SYMB s;
              s = (yyvsp[(2) - (2)].hash_entry)->id_attr;
	      if (s)
		   s->attr = (yyvsp[(1) - (2)].token);
	      else {
		p = undeftype ? global_unknown : impltype[*(yyvsp[(2) - (2)].hash_entry)->ident - 'a'];
                s = install_entry((yyvsp[(2) - (2)].hash_entry), SOFT);
		s->attr = (yyvsp[(1) - (2)].token);
                set_type(s, p, LOCAL);
	      }
	      q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, (yyvsp[(2) - (2)].hash_entry)->id_attr);
	      q = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      (yyval.bf_node) = get_bfnd(fi,ATTR_DECL, SMNULL, q, LLNULL, LLNULL);
	    ;}
    break;

  case 148:
#line 1655 "gram1.y"
    { PTR_TYPE p;
	      PTR_LLND q, r;
	      PTR_SYMB s;
	      int att;

	      att = (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1->entry.Template.ll_ptr1->
		    entry.Template.symbol->attr;
              s = (yyvsp[(3) - (3)].hash_entry)->id_attr;
	      if (s)
		   s->attr = att;
	      else {
		p = undeftype ? global_unknown : impltype[*(yyvsp[(3) - (3)].hash_entry)->ident - 'a'];
                s = install_entry((yyvsp[(3) - (3)].hash_entry), SOFT);
		s->attr = att;
                set_type(s, p, LOCAL);
	      }
	      q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, (yyvsp[(3) - (3)].hash_entry)->id_attr);
	      for (r = (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1;
		   r->entry.list.next;
		   r = r->entry.list.next) ;
	      r->entry.list.next = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);

	    ;}
    break;

  case 149:
#line 1681 "gram1.y"
    { (yyval.token) = ATT_GLOBAL; ;}
    break;

  case 150:
#line 1683 "gram1.y"
    { (yyval.token) = ATT_CLUSTER; ;}
    break;

  case 151:
#line 1695 "gram1.y"
    {
/*		  varleng = ($1<0 || $1==TYLONG ? 0 : typesize[$1]); */
		  vartype = (yyvsp[(1) - (1)].data_type);
		;}
    break;

  case 152:
#line 1702 "gram1.y"
    { (yyval.data_type) = global_int; ;}
    break;

  case 153:
#line 1703 "gram1.y"
    { (yyval.data_type) = global_float; ;}
    break;

  case 154:
#line 1704 "gram1.y"
    { (yyval.data_type) = global_complex; ;}
    break;

  case 155:
#line 1705 "gram1.y"
    { (yyval.data_type) = global_double; ;}
    break;

  case 156:
#line 1706 "gram1.y"
    { (yyval.data_type) = global_dcomplex; ;}
    break;

  case 157:
#line 1707 "gram1.y"
    { (yyval.data_type) = global_bool; ;}
    break;

  case 158:
#line 1708 "gram1.y"
    { (yyval.data_type) = global_string; ;}
    break;

  case 159:
#line 1713 "gram1.y"
    { (yyval.ll_node) = LLNULL; ;}
    break;

  case 160:
#line 1715 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 161:
#line 1719 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, LEN_OP, (yyvsp[(3) - (5)].ll_node), LLNULL, SMNULL); ;}
    break;

  case 162:
#line 1721 "gram1.y"
    { PTR_LLND l;

                 l = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL); 
                 l->entry.string_val = (char *)"*";
                 (yyval.ll_node) = make_llnd(fi, LEN_OP, l,l, SMNULL);
                ;}
    break;

  case 163:
#line 1728 "gram1.y"
    {(yyval.ll_node) = make_llnd(fi, LEN_OP, (yyvsp[(5) - (6)].ll_node), (yyvsp[(5) - (6)].ll_node), SMNULL);;}
    break;

  case 164:
#line 1732 "gram1.y"
    { (yyval.ll_node) = LLNULL; ;}
    break;

  case 165:
#line 1734 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 166:
#line 1736 "gram1.y"
    { /*$$ = make_llnd(fi, PAREN_OP, $2, LLNULL, SMNULL);*/  (yyval.ll_node) = (yyvsp[(3) - (5)].ll_node);  ;}
    break;

  case 167:
#line 1744 "gram1.y"
    { if((yyvsp[(7) - (9)].ll_node)->variant==LENGTH_OP && (yyvsp[(3) - (9)].ll_node)->variant==(yyvsp[(7) - (9)].ll_node)->variant)
                (yyvsp[(7) - (9)].ll_node)->variant=KIND_OP;
                (yyval.ll_node) = make_llnd(fi, CONS, (yyvsp[(3) - (9)].ll_node), (yyvsp[(7) - (9)].ll_node), SMNULL); 
            ;}
    break;

  case 168:
#line 1751 "gram1.y"
    { if(vartype->variant == T_STRING)
                (yyval.ll_node) = make_llnd(fi,LENGTH_OP,(yyvsp[(1) - (1)].ll_node),LLNULL,SMNULL);
              else
                (yyval.ll_node) = make_llnd(fi,KIND_OP,(yyvsp[(1) - (1)].ll_node),LLNULL,SMNULL);
            ;}
    break;

  case 169:
#line 1757 "gram1.y"
    { PTR_LLND l;
	      l = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
	      l->entry.string_val = (char *)"*";
              (yyval.ll_node) = make_llnd(fi,LENGTH_OP,l,LLNULL,SMNULL);
            ;}
    break;

  case 170:
#line 1763 "gram1.y"
    { /* $$ = make_llnd(fi, SPEC_PAIR, $2, LLNULL, SMNULL); */
	     char *q;
             q = (yyvsp[(1) - (2)].ll_node)->entry.string_val;
  	     if (strcmp(q, "len") == 0)
               (yyval.ll_node) = make_llnd(fi,LENGTH_OP,(yyvsp[(2) - (2)].ll_node),LLNULL,SMNULL);
             else
                (yyval.ll_node) = make_llnd(fi,KIND_OP,(yyvsp[(2) - (2)].ll_node),LLNULL,SMNULL);              
            ;}
    break;

  case 171:
#line 1772 "gram1.y"
    { PTR_LLND l;
	      l = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
	      l->entry.string_val = (char *)"*";
              (yyval.ll_node) = make_llnd(fi,LENGTH_OP,l,LLNULL,SMNULL);
            ;}
    break;

  case 172:
#line 1780 "gram1.y"
    {endioctl();;}
    break;

  case 173:
#line 1793 "gram1.y"
    { (yyval.ll_node) = LLNULL; ;}
    break;

  case 174:
#line 1795 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(2) - (2)].ll_node); ;}
    break;

  case 175:
#line 1798 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, POINTST_OP, LLNULL, (yyvsp[(2) - (2)].ll_node), SMNULL); ;}
    break;

  case 176:
#line 1802 "gram1.y"
    { PTR_SYMB s;
	      PTR_LLND q, r;
	      if(! (yyvsp[(5) - (5)].ll_node)) {
		err("No dimensions in DIMENSION statement", 42);
	      }
              if(statement_kind == 1) /*DVM-directive*/
                err("No shape specification", 65);                
	      s = make_array((yyvsp[(4) - (5)].hash_entry), TYNULL, (yyvsp[(5) - (5)].ll_node), ndim, LOCAL);
	      s->attr = s->attr | DIMENSION_BIT;
	      q = make_llnd(fi,ARRAY_REF, (yyvsp[(5) - (5)].ll_node), LLNULL, s);
	      s->type->entry.ar_decl.ranges = (yyvsp[(5) - (5)].ll_node);
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      (yyval.bf_node) = get_bfnd(fi,DIM_STAT, SMNULL, r, LLNULL, LLNULL);
	    ;}
    break;

  case 177:
#line 1817 "gram1.y"
    {  PTR_SYMB s;
	      PTR_LLND q, r;
	      if(! (yyvsp[(4) - (4)].ll_node)) {
		err("No dimensions in DIMENSION statement", 42);
	      }
	      s = make_array((yyvsp[(3) - (4)].hash_entry), TYNULL, (yyvsp[(4) - (4)].ll_node), ndim, LOCAL);
	      s->attr = s->attr | DIMENSION_BIT;
	      q = make_llnd(fi,ARRAY_REF, (yyvsp[(4) - (4)].ll_node), LLNULL, s);
	      s->type->entry.ar_decl.ranges = (yyvsp[(4) - (4)].ll_node);
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      add_to_lowLevelList(r, (yyvsp[(1) - (4)].bf_node)->entry.Template.ll_ptr1);
	;}
    break;

  case 178:
#line 1833 "gram1.y"
    {/* PTR_SYMB s;*/
	      PTR_LLND r;

	         /*if(!$5) {
		   err("No dimensions in ALLOCATABLE statement",305);		
	           }
	          s = make_array($4, TYNULL, $5, ndim, LOCAL);
	          s->attr = s->attr | ALLOCATABLE_BIT;
	          q = make_llnd(fi,ARRAY_REF, $5, LLNULL, s);
	          s->type->entry.ar_decl.ranges = $5;
                  r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
                */
              (yyvsp[(4) - (4)].ll_node)->entry.Template.symbol->attr = (yyvsp[(4) - (4)].ll_node)->entry.Template.symbol->attr | ALLOCATABLE_BIT;
	      r = make_llnd(fi,EXPR_LIST, (yyvsp[(4) - (4)].ll_node), LLNULL, SMNULL);
	      (yyval.bf_node) = get_bfnd(fi, ALLOCATABLE_STMT, SMNULL, r, LLNULL, LLNULL);
	    ;}
    break;

  case 179:
#line 1851 "gram1.y"
    {  /*PTR_SYMB s;*/
	      PTR_LLND r;

	        /*  if(! $4) {
		      err("No dimensions in ALLOCATABLE statement",305);
		
	            }
	           s = make_array($3, TYNULL, $4, ndim, LOCAL);
	           s->attr = s->attr | ALLOCATABLE_BIT;
	           q = make_llnd(fi,ARRAY_REF, $4, LLNULL, s);
	           s->type->entry.ar_decl.ranges = $4;
	           r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
                */
              (yyvsp[(3) - (3)].ll_node)->entry.Template.symbol->attr = (yyvsp[(3) - (3)].ll_node)->entry.Template.symbol->attr | ALLOCATABLE_BIT;
              r = make_llnd(fi,EXPR_LIST, (yyvsp[(3) - (3)].ll_node), LLNULL, SMNULL);
	      add_to_lowLevelList(r, (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);
	;}
    break;

  case 180:
#line 1871 "gram1.y"
    { PTR_SYMB s;
	      PTR_LLND  r;
           
	          /*  if(! $5) {
		      err("No dimensions in POINTER statement",306);	    
	              } 
	             s = make_array($4, TYNULL, $5, ndim, LOCAL);
	             s->attr = s->attr | POINTER_BIT;
	             q = make_llnd(fi,ARRAY_REF, $5, LLNULL, s);
	             s->type->entry.ar_decl.ranges = $5;
                   */

                  /*s = make_pointer( $4->entry.Template.symbol->parent, TYNULL, LOCAL);*/ /*17.02.03*/
                 /*$4->entry.Template.symbol->attr = $4->entry.Template.symbol->attr | POINTER_BIT;*/
              s = (yyvsp[(4) - (4)].ll_node)->entry.Template.symbol; /*17.02.03*/
              s->attr = s->attr | POINTER_BIT;
	      r = make_llnd(fi,EXPR_LIST, (yyvsp[(4) - (4)].ll_node), LLNULL, SMNULL);
	      (yyval.bf_node) = get_bfnd(fi, POINTER_STMT, SMNULL, r, LLNULL, LLNULL);
	    ;}
    break;

  case 181:
#line 1891 "gram1.y"
    {  PTR_SYMB s;
	      PTR_LLND r;

     	        /*  if(! $4) {
	        	err("No dimensions in POINTER statement",306);
	            }
	           s = make_array($3, TYNULL, $4, ndim, LOCAL);
	           s->attr = s->attr | POINTER_BIT;
	           q = make_llnd(fi,ARRAY_REF, $4, LLNULL, s);
	           s->type->entry.ar_decl.ranges = $4;
                */

                /*s = make_pointer( $3->entry.Template.symbol->parent, TYNULL, LOCAL);*/ /*17.02.03*/
                /*$3->entry.Template.symbol->attr = $3->entry.Template.symbol->attr | POINTER_BIT;*/
              s = (yyvsp[(3) - (3)].ll_node)->entry.Template.symbol; /*17.02.03*/
              s->attr = s->attr | POINTER_BIT;
	      r = make_llnd(fi,EXPR_LIST, (yyvsp[(3) - (3)].ll_node), LLNULL, SMNULL);
	      add_to_lowLevelList(r, (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);
	;}
    break;

  case 182:
#line 1913 "gram1.y"
    {/* PTR_SYMB s;*/
	      PTR_LLND r;


	     /* if(! $5) {
		err("No dimensions in TARGET statement",307);
	      }
	      s = make_array($4, TYNULL, $5, ndim, LOCAL);
	      s->attr = s->attr | TARGET_BIT;
	      q = make_llnd(fi,ARRAY_REF, $5, LLNULL, s);
	      s->type->entry.ar_decl.ranges = $5;
             */
              (yyvsp[(4) - (4)].ll_node)->entry.Template.symbol->attr = (yyvsp[(4) - (4)].ll_node)->entry.Template.symbol->attr | TARGET_BIT;
	      r = make_llnd(fi,EXPR_LIST, (yyvsp[(4) - (4)].ll_node), LLNULL, SMNULL);
	      (yyval.bf_node) = get_bfnd(fi, TARGET_STMT, SMNULL, r, LLNULL, LLNULL);
	    ;}
    break;

  case 183:
#line 1930 "gram1.y"
    {  /*PTR_SYMB s;*/
	      PTR_LLND r;

	     /* if(! $4) {
		err("No dimensions in TARGET statement",307);
	      }
	      s = make_array($3, TYNULL, $4, ndim, LOCAL);
	      s->attr = s->attr | TARGET_BIT;
	      q = make_llnd(fi,ARRAY_REF, $4, LLNULL, s);
	      s->type->entry.ar_decl.ranges = $4;
              */
              (yyvsp[(3) - (3)].ll_node)->entry.Template.symbol->attr = (yyvsp[(3) - (3)].ll_node)->entry.Template.symbol->attr | TARGET_BIT;
	      r = make_llnd(fi,EXPR_LIST, (yyvsp[(3) - (3)].ll_node), LLNULL, SMNULL);
	      add_to_lowLevelList(r, (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);
	;}
    break;

  case 184:
#line 1948 "gram1.y"
    { PTR_LLND p, q;

              p = make_llnd(fi,EXPR_LIST, (yyvsp[(3) - (3)].ll_node), LLNULL, SMNULL);
	      q = make_llnd(fi,COMM_LIST, p, LLNULL, SMNULL);
	      (yyval.bf_node) = get_bfnd(fi,COMM_STAT, SMNULL, q, LLNULL, LLNULL);
	    ;}
    break;

  case 185:
#line 1955 "gram1.y"
    { PTR_LLND p, q;

              p = make_llnd(fi,EXPR_LIST, (yyvsp[(4) - (4)].ll_node), LLNULL, SMNULL);
	      q = make_llnd(fi,COMM_LIST, p, LLNULL, (yyvsp[(3) - (4)].symbol));
	      (yyval.bf_node) = get_bfnd(fi,COMM_STAT, SMNULL, q, LLNULL, LLNULL);
	    ;}
    break;

  case 186:
#line 1962 "gram1.y"
    { PTR_LLND p, q;

              p = make_llnd(fi,EXPR_LIST, (yyvsp[(5) - (5)].ll_node), LLNULL, SMNULL);
	      q = make_llnd(fi,COMM_LIST, p, LLNULL, (yyvsp[(3) - (5)].symbol));
	      add_to_lowList(q, (yyvsp[(1) - (5)].bf_node)->entry.Template.ll_ptr1);
	    ;}
    break;

  case 187:
#line 1969 "gram1.y"
    { PTR_LLND p, r;

              p = make_llnd(fi,EXPR_LIST, (yyvsp[(3) - (3)].ll_node), LLNULL, SMNULL);
	      /*q = make_llnd(fi,COMM_LIST, p, LLNULL, SMNULL);*/
	      for (r = (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1;
		   r->entry.list.next;
		   r = r->entry.list.next);
	      add_to_lowLevelList(p, r->entry.Template.ll_ptr1);
	    ;}
    break;

  case 188:
#line 1982 "gram1.y"
    { PTR_LLND q, r;

              q = make_llnd(fi,EXPR_LIST, (yyvsp[(4) - (4)].ll_node), LLNULL, SMNULL);
	      r = make_llnd(fi,NAMELIST_LIST, q, LLNULL, (yyvsp[(3) - (4)].symbol));
	      (yyval.bf_node) = get_bfnd(fi,NAMELIST_STAT, SMNULL, r, LLNULL, LLNULL);
	    ;}
    break;

  case 189:
#line 1989 "gram1.y"
    { PTR_LLND q, r;

              q = make_llnd(fi,EXPR_LIST, (yyvsp[(5) - (5)].ll_node), LLNULL, SMNULL);
	      r = make_llnd(fi,NAMELIST_LIST, q, LLNULL, (yyvsp[(3) - (5)].symbol));
	      add_to_lowList(r, (yyvsp[(1) - (5)].bf_node)->entry.Template.ll_ptr1);
	    ;}
    break;

  case 190:
#line 1996 "gram1.y"
    { PTR_LLND q, r;

              q = make_llnd(fi,EXPR_LIST, (yyvsp[(3) - (3)].ll_node), LLNULL, SMNULL);
	      for (r = (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1;
		   r->entry.list.next;
		   r = r->entry.list.next);
	      add_to_lowLevelList(q, r->entry.Template.ll_ptr1);
	    ;}
    break;

  case 191:
#line 2007 "gram1.y"
    { (yyval.symbol) =  make_local_entity((yyvsp[(2) - (3)].hash_entry), NAMELIST_NAME,global_default,LOCAL); ;}
    break;

  case 192:
#line 2011 "gram1.y"
    { (yyval.symbol) = NULL; /*make_common(look_up_sym("*"));*/ ;}
    break;

  case 193:
#line 2013 "gram1.y"
    { (yyval.symbol) = make_common((yyvsp[(2) - (3)].hash_entry)); ;}
    break;

  case 194:
#line 2018 "gram1.y"
    {  PTR_SYMB s;
	
	      if((yyvsp[(2) - (2)].ll_node)) {
		s = make_array((yyvsp[(1) - (2)].hash_entry), TYNULL, (yyvsp[(2) - (2)].ll_node), ndim, LOCAL);
                s->attr = s->attr | DIMENSION_BIT;
		s->type->entry.ar_decl.ranges = (yyvsp[(2) - (2)].ll_node);
		(yyval.ll_node) = make_llnd(fi,ARRAY_REF, (yyvsp[(2) - (2)].ll_node), LLNULL, s);
	      }
	      else {
		s = make_scalar((yyvsp[(1) - (2)].hash_entry), TYNULL, LOCAL);	
		(yyval.ll_node) = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
	      }

          ;}
    break;

  case 195:
#line 2036 "gram1.y"
    { PTR_LLND p, q;
              PTR_SYMB s;

	      s = make_external((yyvsp[(4) - (4)].hash_entry), TYNULL);
	      s->attr = s->attr | EXTERNAL_BIT;
              q = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
	      p = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      (yyval.bf_node) = get_bfnd(fi,EXTERN_STAT, SMNULL, p, LLNULL, LLNULL);
	    ;}
    break;

  case 196:
#line 2047 "gram1.y"
    { PTR_LLND p, q;
              PTR_SYMB s;

	      s = make_external((yyvsp[(3) - (3)].hash_entry), TYNULL);
	      s->attr = s->attr | EXTERNAL_BIT;
              p = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
	      q = make_llnd(fi,EXPR_LIST, p, LLNULL, SMNULL);
	      add_to_lowLevelList(q, (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);
	    ;}
    break;

  case 197:
#line 2059 "gram1.y"
    { PTR_LLND p, q;
              PTR_SYMB s;

	      s = make_intrinsic((yyvsp[(4) - (4)].hash_entry), TYNULL); /*make_function($3, TYNULL, NO);*/
	      s->attr = s->attr | INTRINSIC_BIT;
              q = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
	      p = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      (yyval.bf_node) = get_bfnd(fi,INTRIN_STAT, SMNULL, p,
			     LLNULL, LLNULL);
	    ;}
    break;

  case 198:
#line 2071 "gram1.y"
    { PTR_LLND p, q;
              PTR_SYMB s;

	      s = make_intrinsic((yyvsp[(3) - (3)].hash_entry), TYNULL); /* make_function($3, TYNULL, NO);*/
	      s->attr = s->attr | INTRINSIC_BIT;
              p = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
	      q = make_llnd(fi,EXPR_LIST, p, LLNULL, SMNULL);
	      add_to_lowLevelList(q, (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);
	    ;}
    break;

  case 199:
#line 2085 "gram1.y"
    {
	      (yyval.bf_node) = get_bfnd(fi,EQUI_STAT, SMNULL, (yyvsp[(3) - (3)].ll_node),
			     LLNULL, LLNULL);
	    ;}
    break;

  case 200:
#line 2091 "gram1.y"
    { 
	      add_to_lowLevelList((yyvsp[(3) - (3)].ll_node), (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);
	    ;}
    break;

  case 201:
#line 2098 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,EQUI_LIST, (yyvsp[(2) - (3)].ll_node), LLNULL, SMNULL);
           ;}
    break;

  case 202:
#line 2104 "gram1.y"
    { PTR_LLND p;
	      p = make_llnd(fi,EXPR_LIST, (yyvsp[(3) - (3)].ll_node), LLNULL, SMNULL);
	      (yyval.ll_node) = make_llnd(fi,EXPR_LIST, (yyvsp[(1) - (3)].ll_node), p, SMNULL);
	    ;}
    break;

  case 203:
#line 2110 "gram1.y"
    { PTR_LLND p;

	      p = make_llnd(fi,EXPR_LIST, (yyvsp[(3) - (3)].ll_node), LLNULL, SMNULL);
	      add_to_lowLevelList(p, (yyvsp[(1) - (3)].ll_node));
	    ;}
    break;

  case 204:
#line 2118 "gram1.y"
    {  PTR_SYMB s;
           s=make_scalar((yyvsp[(1) - (1)].hash_entry),TYNULL,LOCAL);
           (yyval.ll_node) = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
           s->attr = s->attr | EQUIVALENCE_BIT;
            /*$$=$1; $$->entry.Template.symbol->attr = $$->entry.Template.symbol->attr | EQUIVALENCE_BIT; */
        ;}
    break;

  case 205:
#line 2125 "gram1.y"
    {  PTR_SYMB s;
           s=make_array((yyvsp[(1) - (4)].hash_entry),TYNULL,LLNULL,0,LOCAL);
           (yyval.ll_node) = make_llnd(fi,ARRAY_REF, (yyvsp[(3) - (4)].ll_node), LLNULL, s);
           s->attr = s->attr | EQUIVALENCE_BIT;
            /*$$->entry.Template.symbol->attr = $$->entry.Template.symbol->attr | EQUIVALENCE_BIT; */
        ;}
    break;

  case 207:
#line 2144 "gram1.y"
    { PTR_LLND p;
              data_stat = NO;
	      p = make_llnd(fi,STMT_STR, LLNULL, LLNULL,
			    SMNULL);
              p->entry.string_val = copys(stmtbuf);
	      (yyval.bf_node) = get_bfnd(fi,DATA_DECL, SMNULL, p, LLNULL, LLNULL);
            ;}
    break;

  case 210:
#line 2158 "gram1.y"
    {data_stat = YES;;}
    break;

  case 211:
#line 2162 "gram1.y"
    {
	      if (parstate == OUTSIDE)
	         { PTR_BFND p;

		   p = get_bfnd(fi,PROG_HEDR,
                                make_program(look_up_sym("_MAIN")),
                                LLNULL, LLNULL, LLNULL);
		   set_blobs(p, global_bfnd, NEW_GROUP1);
	           add_scope_level(p, NO);
		   position = IN_PROC; 
	  	   /*parstate = INDCL;*/
                 }
	      if(parstate < INDCL)
		{
		  /* enddcl();*/
		  parstate = INDCL;
		}
	    ;}
    break;

  case 222:
#line 2207 "gram1.y"
    {;;}
    break;

  case 223:
#line 2211 "gram1.y"
    { (yyval.symbol)= make_scalar((yyvsp[(1) - (1)].hash_entry), TYNULL, LOCAL);;}
    break;

  case 224:
#line 2215 "gram1.y"
    { (yyval.symbol)= make_scalar((yyvsp[(1) - (1)].hash_entry), TYNULL, LOCAL); 
              (yyval.symbol)->attr = (yyval.symbol)->attr | DATA_BIT; 
            ;}
    break;

  case 225:
#line 2221 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, DATA_SUBS, (yyvsp[(2) - (3)].ll_node), LLNULL, SMNULL); ;}
    break;

  case 226:
#line 2225 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, DATA_RANGE, (yyvsp[(2) - (5)].ll_node), (yyvsp[(4) - (5)].ll_node), SMNULL); ;}
    break;

  case 227:
#line 2229 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 228:
#line 2231 "gram1.y"
    { (yyval.ll_node) = add_to_lowLevelList((yyvsp[(3) - (3)].ll_node), (yyvsp[(1) - (3)].ll_node)); ;}
    break;

  case 229:
#line 2235 "gram1.y"
    { (yyval.ll_node) = LLNULL; ;}
    break;

  case 230:
#line 2237 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 231:
#line 2241 "gram1.y"
    {(yyval.ll_node)= make_llnd(fi, DATA_IMPL_DO, (yyvsp[(2) - (7)].ll_node), (yyvsp[(6) - (7)].ll_node), (yyvsp[(4) - (7)].symbol)); ;}
    break;

  case 232:
#line 2245 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 233:
#line 2247 "gram1.y"
    { (yyval.ll_node) = add_to_lowLevelList((yyvsp[(3) - (3)].ll_node), (yyvsp[(1) - (3)].ll_node)); ;}
    break;

  case 234:
#line 2251 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, DATA_ELT, (yyvsp[(2) - (2)].ll_node), LLNULL, (yyvsp[(1) - (2)].symbol)); ;}
    break;

  case 235:
#line 2253 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, DATA_ELT, (yyvsp[(2) - (2)].ll_node), LLNULL, (yyvsp[(1) - (2)].symbol)); ;}
    break;

  case 236:
#line 2255 "gram1.y"
    {
              (yyvsp[(2) - (3)].ll_node)->entry.Template.ll_ptr2 = (yyvsp[(3) - (3)].ll_node);
              (yyval.ll_node) = make_llnd(fi, DATA_ELT, (yyvsp[(2) - (3)].ll_node), LLNULL, (yyvsp[(1) - (3)].symbol)); 
            ;}
    break;

  case 237:
#line 2260 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, DATA_ELT, (yyvsp[(1) - (1)].ll_node), LLNULL, SMNULL); ;}
    break;

  case 251:
#line 2284 "gram1.y"
    {if((yyvsp[(2) - (6)].ll_node)->entry.Template.symbol->variant != TYPE_NAME)
               errstr("Undefined type %s",(yyvsp[(2) - (6)].ll_node)->entry.Template.symbol->ident,319); 
           ;}
    break;

  case 268:
#line 2329 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ICON_EXPR, (yyvsp[(1) - (1)].ll_node), LLNULL, SMNULL); ;}
    break;

  case 269:
#line 2331 "gram1.y"
    {
              PTR_LLND p;

              p = intrinsic_op_node("+", UNARY_ADD_OP, (yyvsp[(2) - (2)].ll_node), LLNULL);
              (yyval.ll_node) = make_llnd(fi,ICON_EXPR, p, LLNULL, SMNULL);
            ;}
    break;

  case 270:
#line 2338 "gram1.y"
    {
              PTR_LLND p;
 
              p = intrinsic_op_node("-", MINUS_OP, (yyvsp[(2) - (2)].ll_node), LLNULL);
              (yyval.ll_node) = make_llnd(fi,ICON_EXPR, p, LLNULL, SMNULL);
            ;}
    break;

  case 271:
#line 2345 "gram1.y"
    {
              PTR_LLND p;
 
              p = intrinsic_op_node("+", ADD_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node));
              (yyval.ll_node) = make_llnd(fi,ICON_EXPR, p, LLNULL, SMNULL);
            ;}
    break;

  case 272:
#line 2352 "gram1.y"
    {
              PTR_LLND p;
 
              p = intrinsic_op_node("-", SUBT_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node));
              (yyval.ll_node) = make_llnd(fi,ICON_EXPR, p, LLNULL, SMNULL);
            ;}
    break;

  case 273:
#line 2361 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 274:
#line 2363 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node("*", MULT_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 275:
#line 2365 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node("/", DIV_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 276:
#line 2369 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 277:
#line 2371 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node("**", EXP_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 278:
#line 2375 "gram1.y"
    {
              PTR_LLND p;

              p = make_llnd(fi,INT_VAL, LLNULL, LLNULL, SMNULL);
              p->entry.ival = atoi(yytext);
              p->type = global_int;
              (yyval.ll_node) = make_llnd(fi,EXPR_LIST, p, LLNULL, SMNULL);
            ;}
    break;

  case 279:
#line 2384 "gram1.y"
    {
              PTR_LLND p;
 
              p = make_llnd(fi,VAR_REF, LLNULL, LLNULL, (yyvsp[(1) - (1)].symbol));
              (yyval.ll_node) = make_llnd(fi,EXPR_LIST, p, LLNULL, SMNULL);
            ;}
    break;

  case 280:
#line 2391 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,EXPR_LIST, (yyvsp[(2) - (3)].ll_node), LLNULL, SMNULL);
            ;}
    break;

  case 281:
#line 2398 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,EXPR_LIST, (yyvsp[(1) - (1)].ll_node), LLNULL, SMNULL); ;}
    break;

  case 282:
#line 2400 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); ;}
    break;

  case 283:
#line 2404 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);
             (yyval.ll_node)->entry.Template.symbol->attr = (yyval.ll_node)->entry.Template.symbol->attr | SAVE_BIT;
           ;}
    break;

  case 284:
#line 2408 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,COMM_LIST, LLNULL, LLNULL, (yyvsp[(1) - (1)].symbol)); 
            (yyval.ll_node)->entry.Template.symbol->attr = (yyval.ll_node)->entry.Template.symbol->attr | SAVE_BIT;
          ;}
    break;

  case 285:
#line 2414 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(2) - (3)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 286:
#line 2416 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (5)].ll_node), (yyvsp[(4) - (5)].ll_node), EXPR_LIST); ;}
    break;

  case 287:
#line 2420 "gram1.y"
    { as_op_kwd_ = YES; ;}
    break;

  case 288:
#line 2424 "gram1.y"
    { as_op_kwd_ = NO; ;}
    break;

  case 289:
#line 2429 "gram1.y"
    { 
             PTR_SYMB s; 
             s = make_scalar((yyvsp[(1) - (1)].hash_entry), TYNULL, LOCAL);	
	     s->attr = s->attr | type_opt;
	     (yyval.ll_node) = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
            ;}
    break;

  case 290:
#line 2436 "gram1.y"
    { PTR_SYMB s;
	      s = make_function((yyvsp[(3) - (4)].hash_entry), global_default, LOCAL);
	      s->variant = INTERFACE_NAME;
              s->attr = s->attr | type_opt;
              (yyval.ll_node) = make_llnd(fi,OPERATOR_OP, LLNULL, LLNULL, s);
	    ;}
    break;

  case 291:
#line 2443 "gram1.y"
    { PTR_SYMB s;
	      s = make_procedure(look_up_sym("="), LOCAL);
	      s->variant = INTERFACE_NAME;
              s->attr = s->attr | type_opt;
              (yyval.ll_node) = make_llnd(fi,ASSIGNMENT_OP, LLNULL, LLNULL, s);
	    ;}
    break;

  case 292:
#line 2453 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 293:
#line 2455 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); ;}
    break;

  case 294:
#line 2459 "gram1.y"
    { PTR_SYMB p;

                /* The check if name and expr have compatible types has
                   not been done yet. */ 
		p = make_constant((yyvsp[(1) - (3)].hash_entry), TYNULL);
 	        p->attr = p->attr | PARAMETER_BIT;
                p->entry.const_value = (yyvsp[(3) - (3)].ll_node);
		(yyval.ll_node) = make_llnd(fi,CONST_REF, LLNULL, LLNULL, p);
	    ;}
    break;

  case 295:
#line 2471 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, MODULE_PROC_STMT, SMNULL, (yyvsp[(2) - (2)].ll_node), LLNULL, LLNULL); ;}
    break;

  case 296:
#line 2474 "gram1.y"
    { PTR_SYMB s;
 	      PTR_LLND q;

	      s = make_function((yyvsp[(1) - (1)].hash_entry), TYNULL, LOCAL);
	      s->variant = ROUTINE_NAME;
              q = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
	      (yyval.ll_node) = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	    ;}
    break;

  case 297:
#line 2483 "gram1.y"
    { PTR_LLND p, q;
              PTR_SYMB s;

	      s = make_function((yyvsp[(3) - (3)].hash_entry), TYNULL, LOCAL);
	      s->variant = ROUTINE_NAME;
              p = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
	      q = make_llnd(fi,EXPR_LIST, p, LLNULL, SMNULL);
	      add_to_lowLevelList(q, (yyvsp[(1) - (3)].ll_node));
	    ;}
    break;

  case 298:
#line 2496 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, USE_STMT, (yyvsp[(3) - (3)].symbol), LLNULL, LLNULL, LLNULL);
              /*add_scope_level($3->entry.Template.func_hedr, YES);*/ /*17.06.01*/
              copy_module_scope((yyvsp[(3) - (3)].symbol),LLNULL); /*17.03.03*/
              colon_flag = NO;
            ;}
    break;

  case 299:
#line 2502 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, USE_STMT, (yyvsp[(3) - (6)].symbol), (yyvsp[(6) - (6)].ll_node), LLNULL, LLNULL); 
              /*add_scope_level(module_scope, YES); *//* 17.06.01*/
              copy_module_scope((yyvsp[(3) - (6)].symbol),(yyvsp[(6) - (6)].ll_node)); /*17.03.03 */
              colon_flag = NO;
            ;}
    break;

  case 300:
#line 2508 "gram1.y"
    { PTR_LLND l;

	      l = make_llnd(fi, ONLY_NODE, LLNULL, LLNULL, SMNULL);
              (yyval.bf_node) = get_bfnd(fi, USE_STMT, (yyvsp[(3) - (6)].symbol), l, LLNULL, LLNULL);
            ;}
    break;

  case 301:
#line 2514 "gram1.y"
    { PTR_LLND l;

	      l = make_llnd(fi, ONLY_NODE, (yyvsp[(7) - (7)].ll_node), LLNULL, SMNULL);
              (yyval.bf_node) = get_bfnd(fi, USE_STMT, (yyvsp[(3) - (7)].symbol), l, LLNULL, LLNULL);
            ;}
    break;

  case 302:
#line 2522 "gram1.y"
    {
              if ((yyvsp[(1) - (1)].hash_entry)->id_attr == SMNULL)
	         warn1("Unknown module %s", (yyvsp[(1) - (1)].hash_entry)->ident,308);
              (yyval.symbol) = make_global_entity((yyvsp[(1) - (1)].hash_entry), MODULE_NAME, global_default, NO);
	      module_scope = (yyval.symbol)->entry.Template.func_hedr;
           
            ;}
    break;

  case 303:
#line 2532 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 304:
#line 2534 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); ;}
    break;

  case 305:
#line 2538 "gram1.y"
    {  (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 306:
#line 2540 "gram1.y"
    {  PTR_HASH oldhash,copyhash;
	       PTR_SYMB oldsym, newsym;
	       PTR_LLND m;

	       oldhash = just_look_up_sym_in_scope(module_scope, (yyvsp[(1) - (1)].hash_entry)->ident);
	       if (oldhash == HSNULL) {
                  errstr("Unknown identifier %s.", (yyvsp[(1) - (1)].hash_entry)->ident,309);
	          (yyval.ll_node)= LLNULL;
	       }
	       else {
                 oldsym = oldhash->id_attr;
                 copyhash=just_look_up_sym_in_scope(cur_scope(), (yyvsp[(1) - (1)].hash_entry)->ident);
	         if( copyhash && copyhash->id_attr && copyhash->id_attr->entry.Template.tag==module_scope->id)
                 {
                   newsym = copyhash->id_attr;
                   newsym->entry.Template.tag = 0;
                 }
                 else
                 {
	           newsym = make_local_entity((yyvsp[(1) - (1)].hash_entry), oldsym->variant, oldsym->type,LOCAL);
	           /* copies data in entry.Template structure and attr */
	           copy_sym_data(oldsym, newsym);	         
	             /*newsym->entry.Template.base_name = oldsym;*//*19.03.03*/
                 }
	  	/* l = make_llnd(fi, VAR_REF, LLNULL, LLNULL, oldsym);*/
		 m = make_llnd(fi, VAR_REF, LLNULL, LLNULL, newsym);
		 (yyval.ll_node) = make_llnd(fi, RENAME_NODE, m, LLNULL, oldsym);
 	      }
            ;}
    break;

  case 307:
#line 2573 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 308:
#line 2575 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); ;}
    break;

  case 309:
#line 2579 "gram1.y"
    {  PTR_HASH oldhash,copyhash;
	       PTR_SYMB oldsym, newsym;
	       PTR_LLND l, m;

	       oldhash = just_look_up_sym_in_scope(module_scope, (yyvsp[(3) - (3)].hash_entry)->ident);
	       if (oldhash == HSNULL) {
                  errstr("Unknown identifier %s", (yyvsp[(3) - (3)].hash_entry)->ident,309);
	          (yyval.ll_node)= LLNULL;
	       }
	       else {
                 oldsym = oldhash->id_attr;
                 copyhash = just_look_up_sym_in_scope(cur_scope(), (yyvsp[(3) - (3)].hash_entry)->ident);
	         if(copyhash && copyhash->id_attr && copyhash->id_attr->entry.Template.tag==module_scope->id)
                 {
                    delete_symbol(copyhash->id_attr);
                    copyhash->id_attr = SMNULL;
                 }
                   newsym = make_local_entity((yyvsp[(1) - (3)].hash_entry), oldsym->variant, oldsym->type, LOCAL);
	           /* copies data in entry.Template structure and attr */
	           copy_sym_data(oldsym, newsym);	
                         
	           /*newsym->entry.Template.base_name = oldsym;*//*19.03.03*/
	  	 l  = make_llnd(fi, VAR_REF, LLNULL, LLNULL, oldsym);
		 m  = make_llnd(fi, VAR_REF, LLNULL, LLNULL, newsym);
		 (yyval.ll_node) = make_llnd(fi, RENAME_NODE, m, l, SMNULL);
 	      }
            ;}
    break;

  case 310:
#line 2617 "gram1.y"
    { ndim = 0;	explicit_shape = 1; (yyval.ll_node) = LLNULL; ;}
    break;

  case 311:
#line 2619 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(2) - (3)].ll_node); ;}
    break;

  case 312:
#line 2622 "gram1.y"
    { ndim = 0; explicit_shape = 1;;}
    break;

  case 313:
#line 2623 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,EXPR_LIST, (yyvsp[(2) - (2)].ll_node), LLNULL, SMNULL);
	      (yyval.ll_node)->type = global_default;
	    ;}
    break;

  case 314:
#line 2628 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); ;}
    break;

  case 315:
#line 2632 "gram1.y"
    {
	      if(ndim == maxdim)
		err("Too many dimensions", 43);
	      else if(ndim < maxdim)
		(yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);
	      ++ndim;
	    ;}
    break;

  case 316:
#line 2640 "gram1.y"
    {
	      if(ndim == maxdim)
		err("Too many dimensions", 43);
	      else if(ndim < maxdim)
		(yyval.ll_node) = make_llnd(fi, DDOT, LLNULL, LLNULL, SMNULL);
	      ++ndim;
              explicit_shape = 0;
	    ;}
    break;

  case 317:
#line 2649 "gram1.y"
    {
	      if(ndim == maxdim)
		err("Too many dimensions", 43);
	      else if(ndim < maxdim)
		(yyval.ll_node) = make_llnd(fi,DDOT, (yyvsp[(1) - (2)].ll_node), LLNULL, SMNULL);
	      ++ndim;
              explicit_shape = 0;
	    ;}
    break;

  case 318:
#line 2658 "gram1.y"
    {
	      if(ndim == maxdim)
		err("Too many dimensions", 43);
	      else if(ndim < maxdim)
		(yyval.ll_node) = make_llnd(fi,DDOT, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), SMNULL);
	      ++ndim;
	    ;}
    break;

  case 319:
#line 2668 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,STAR_RANGE, LLNULL, LLNULL, SMNULL);
	      (yyval.ll_node)->type = global_default;
              explicit_shape = 0;
	    ;}
    break;

  case 321:
#line 2677 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 322:
#line 2679 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); ;}
    break;

  case 323:
#line 2683 "gram1.y"
    {PTR_LABEL p;
	     p = make_label_node(fi,convci(yyleng, yytext));
	     p->scope = cur_scope();
	     (yyval.ll_node) = make_llnd_label(fi,LABEL_REF, p);
	  ;}
    break;

  case 324:
#line 2691 "gram1.y"
    { /*PTR_LLND l;*/

          /*   l = make_llnd(fi, EXPR_LIST, $3, LLNULL, SMNULL);*/
             (yyval.bf_node) = get_bfnd(fi,IMPL_DECL, SMNULL, (yyvsp[(3) - (3)].ll_node), LLNULL, LLNULL);
             redefine_func_arg_type();
           ;}
    break;

  case 325:
#line 2706 "gram1.y"
    { /*undeftype = YES;
	    setimpl(TYNULL, (int)'a', (int)'z'); FB COMMENTED---> NOT QUITE RIGHT BUT AVOID PB WITH COMMON*/
	    (yyval.bf_node) = get_bfnd(fi,IMPL_DECL, SMNULL, LLNULL, LLNULL, LLNULL);
	  ;}
    break;

  case 326:
#line 2713 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 327:
#line 2715 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); ;}
    break;

  case 328:
#line 2719 "gram1.y"
    { 

            (yyval.ll_node) = make_llnd(fi, IMPL_TYPE, (yyvsp[(3) - (4)].ll_node), LLNULL, SMNULL);
            (yyval.ll_node)->type = vartype;
          ;}
    break;

  case 329:
#line 2734 "gram1.y"
    { implkwd = YES; ;}
    break;

  case 330:
#line 2735 "gram1.y"
    { vartype = (yyvsp[(2) - (2)].data_type); ;}
    break;

  case 331:
#line 2739 "gram1.y"
    { (yyval.data_type) = (yyvsp[(2) - (2)].data_type); ;}
    break;

  case 332:
#line 2741 "gram1.y"
    { (yyval.data_type) = (yyvsp[(1) - (1)].data_type);;}
    break;

  case 333:
#line 2753 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 334:
#line 2755 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); ;}
    break;

  case 335:
#line 2759 "gram1.y"
    {
	      setimpl(vartype, (int)(yyvsp[(1) - (1)].charv), (int)(yyvsp[(1) - (1)].charv));
	      (yyval.ll_node) = make_llnd(fi,CHAR_VAL, LLNULL, LLNULL, SMNULL);
	      (yyval.ll_node)->entry.cval = (yyvsp[(1) - (1)].charv);
	    ;}
    break;

  case 336:
#line 2765 "gram1.y"
    { PTR_LLND p,q;
	      
	      setimpl(vartype, (int)(yyvsp[(1) - (3)].charv), (int)(yyvsp[(3) - (3)].charv));
	      p = make_llnd(fi,CHAR_VAL, LLNULL, LLNULL, SMNULL);
	      p->entry.cval = (yyvsp[(1) - (3)].charv);
	      q = make_llnd(fi,CHAR_VAL, LLNULL, LLNULL, SMNULL);
	      q->entry.cval = (yyvsp[(3) - (3)].charv);
	      (yyval.ll_node)= make_llnd(fi,DDOT, p, q, SMNULL);
	    ;}
    break;

  case 337:
#line 2777 "gram1.y"
    {
	      if(yyleng!=1 || yytext[0]<'a' || yytext[0]>'z')
		{
		  err("IMPLICIT item must be single letter", 37);
		  (yyval.charv) = '\0';
		}
	      else (yyval.charv) = yytext[0];
	    ;}
    break;

  case 338:
#line 2788 "gram1.y"
    {
	      if (parstate == OUTSIDE)
	         { PTR_BFND p;

		   p = get_bfnd(fi,PROG_HEDR,
                                make_program(look_up_sym("_MAIN")),
                                LLNULL, LLNULL, LLNULL);
		   set_blobs(p, global_bfnd, NEW_GROUP1);
	           add_scope_level(p, NO);
		   position = IN_PROC; 
	  	   parstate = INSIDE;
                 }
	  
	    ;}
    break;

  case 339:
#line 2805 "gram1.y"
    { switch(parstate)
		{
                case OUTSIDE:  
			{ PTR_BFND p;

			  p = get_bfnd(fi,PROG_HEDR,
                                       make_program(look_up_sym("_MAIN")),
                                       LLNULL, LLNULL, LLNULL);
			  set_blobs(p, global_bfnd, NEW_GROUP1);
			  add_scope_level(p, NO);
			  position = IN_PROC; 
	  		  parstate = INDCL; }
	                  break;
                case INSIDE:    parstate = INDCL;
                case INDCL:     break;

                case INDATA:
                         /*  err(
                     "Statement order error: declaration after DATA or function statement", 
                                 29);*/
                              break;

                default:
                           err("Declaration among executables", 30);
                }
        ;}
    break;

  case 342:
#line 2843 "gram1.y"
    { (yyval.ll_node) = LLNULL; endioctl(); ;}
    break;

  case 343:
#line 2845 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);  endioctl();;}
    break;

  case 344:
#line 2849 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 345:
#line 2851 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 346:
#line 2853 "gram1.y"
    { PTR_LLND l;
	      l = make_llnd(fi, KEYWORD_ARG, (yyvsp[(1) - (2)].ll_node), (yyvsp[(2) - (2)].ll_node), SMNULL);
	      l->type = (yyvsp[(2) - (2)].ll_node)->type;
              (yyval.ll_node) = l; 
	    ;}
    break;

  case 347:
#line 2864 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(2) - (2)].ll_node), LLNULL, EXPR_LIST);
              endioctl(); 
            ;}
    break;

  case 348:
#line 2868 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (4)].ll_node), (yyvsp[(4) - (4)].ll_node), EXPR_LIST);
              endioctl(); 
            ;}
    break;

  case 349:
#line 2874 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 350:
#line 2876 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); ;}
    break;

  case 351:
#line 2880 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 352:
#line 2882 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(2) - (3)].ll_node); ;}
    break;

  case 353:
#line 2884 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 354:
#line 2888 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 355:
#line 2890 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 356:
#line 2894 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 357:
#line 2896 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node("+", ADD_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 358:
#line 2898 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node("-", SUBT_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 359:
#line 2900 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node("*", MULT_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 360:
#line 2902 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node("/", DIV_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 361:
#line 2904 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node("**", EXP_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 362:
#line 2906 "gram1.y"
    { (yyval.ll_node) = defined_op_node((yyvsp[(1) - (2)].hash_entry), (yyvsp[(2) - (2)].ll_node), LLNULL); ;}
    break;

  case 363:
#line 2908 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node("+", UNARY_ADD_OP, (yyvsp[(2) - (2)].ll_node), LLNULL); ;}
    break;

  case 364:
#line 2910 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node("-", MINUS_OP, (yyvsp[(2) - (2)].ll_node), LLNULL); ;}
    break;

  case 365:
#line 2912 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node(".eq.", EQ_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 366:
#line 2914 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node(".gt.", GT_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 367:
#line 2916 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node(".lt.", LT_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 368:
#line 2918 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node(".ge.", GTEQL_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 369:
#line 2920 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node(".ge.", LTEQL_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 370:
#line 2922 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node(".ne.", NOTEQL_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 371:
#line 2924 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node(".eqv.", EQV_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 372:
#line 2926 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node(".neqv.", NEQV_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 373:
#line 2928 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node(".xor.", XOR_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 374:
#line 2930 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node(".or.", OR_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 375:
#line 2932 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node(".and.", AND_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 376:
#line 2934 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node(".not.", NOT_OP, (yyvsp[(2) - (2)].ll_node), LLNULL); ;}
    break;

  case 377:
#line 2936 "gram1.y"
    { (yyval.ll_node) = intrinsic_op_node("//", CONCAT_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 378:
#line 2938 "gram1.y"
    { (yyval.ll_node) = defined_op_node((yyvsp[(2) - (3)].hash_entry), (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node)); ;}
    break;

  case 379:
#line 2941 "gram1.y"
    { (yyval.token) = ADD_OP; ;}
    break;

  case 380:
#line 2942 "gram1.y"
    { (yyval.token) = SUBT_OP; ;}
    break;

  case 381:
#line 2954 "gram1.y"
    { PTR_SYMB s;
	      PTR_TYPE t;
	     /* PTR_LLND l;*/

       	      if (!(s = (yyvsp[(1) - (1)].hash_entry)->id_attr))
              {
	         s = make_scalar((yyvsp[(1) - (1)].hash_entry), TYNULL, LOCAL);
	     	 s->decl = SOFT;
	      } 
	
	      switch (s->variant)
              {
	      case CONST_NAME:
		   (yyval.ll_node) = make_llnd(fi,CONST_REF,LLNULL,LLNULL, s);
		   t = s->type;
	           if ((t != TYNULL) &&
                       ((t->variant == T_ARRAY) ||  (t->variant == T_STRING) ))
                                 (yyval.ll_node)->variant = ARRAY_REF;

                   (yyval.ll_node)->type = t;
	           break;
	      case DEFAULT:   /* if common region with same name has been
                                 declared. */
		   s = make_scalar((yyvsp[(1) - (1)].hash_entry), TYNULL, LOCAL);
	     	   s->decl = SOFT;

	      case VARIABLE_NAME:
                   (yyval.ll_node) = make_llnd(fi,VAR_REF,LLNULL,LLNULL, s);
	           t = s->type;
	           if (t != TYNULL) {
                     if ((t->variant == T_ARRAY) ||  (t->variant == T_STRING) ||
                         ((t->variant == T_POINTER) && (t->entry.Template.base_type->variant == T_ARRAY) ) )
                         (yyval.ll_node)->variant = ARRAY_REF;

/*  	              if (t->variant == T_DERIVED_TYPE)
                         $$->variant = RECORD_REF; */
	           }
                   (yyval.ll_node)->type = t;
	           break;
	      case TYPE_NAME:
  	           (yyval.ll_node) = make_llnd(fi,TYPE_REF,LLNULL,LLNULL, s);
	           (yyval.ll_node)->type = s->type;
	           break;
	      case INTERFACE_NAME:
  	           (yyval.ll_node) = make_llnd(fi, INTERFACE_REF,LLNULL,LLNULL, s);
	           (yyval.ll_node)->type = s->type;
	           break;
              case FUNCTION_NAME:
                   if(isResultVar(s)) {
                     (yyval.ll_node) = make_llnd(fi,VAR_REF,LLNULL,LLNULL, s);
	             t = s->type;
	             if (t != TYNULL) {
                       if ((t->variant == T_ARRAY) ||  (t->variant == T_STRING) ||
                         ((t->variant == T_POINTER) && (t->entry.Template.base_type->variant == T_ARRAY) ) )
                         (yyval.ll_node)->variant = ARRAY_REF;
	             }
                     (yyval.ll_node)->type = t;
	             break;
                   }                                        
	      default:
  	           (yyval.ll_node) = make_llnd(fi,VAR_REF,LLNULL,LLNULL, s);
	           (yyval.ll_node)->type = s->type;
	           break;
	      }
             /* if ($$->variant == T_POINTER) {
	         l = $$;
	         $$ = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	         $$->type = l->type->entry.Template.base_type;
	      }
              */ /*11.02.03*/
           ;}
    break;

  case 382:
#line 3028 "gram1.y"
    { PTR_SYMB  s;
	      (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); 
              s= (yyval.ll_node)->entry.Template.symbol;
              if ((((yyvsp[(1) - (1)].ll_node)->variant == VAR_REF) || ((yyvsp[(1) - (1)].ll_node)->variant == ARRAY_REF))  && (s->scope !=cur_scope()))  /*global_bfnd*/
              {
	          if(((s->variant == FUNCTION_NAME) && (!isResultVar(s))) || (s->variant == PROCEDURE_NAME) || (s->variant == ROUTINE_NAME))
                  { s = (yyval.ll_node)->entry.Template.symbol =  make_scalar(s->parent, TYNULL, LOCAL);
		    (yyval.ll_node)->type = s->type;  
		  }
              }
            ;}
    break;

  case 383:
#line 3040 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 384:
#line 3042 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 385:
#line 3046 "gram1.y"
    { int num_triplets;
	      PTR_SYMB s;  /*, sym;*/
	      /* PTR_LLND l; */
	      PTR_TYPE tp;
	      /* l = $1; */
	      s = (yyvsp[(1) - (5)].ll_node)->entry.Template.symbol;
            
	      /* Handle variable to function conversion. */
	      if (((yyvsp[(1) - (5)].ll_node)->variant == VAR_REF) && 
	          (((s->variant == VARIABLE_NAME) && (s->type) &&
                    (s->type->variant != T_ARRAY)) ||
  	            (s->variant == ROUTINE_NAME))) {
	        s = (yyvsp[(1) - (5)].ll_node)->entry.Template.symbol =  make_function(s->parent, TYNULL, LOCAL);
	        (yyvsp[(1) - (5)].ll_node)->variant = FUNC_CALL;
              }
	      if (((yyvsp[(1) - (5)].ll_node)->variant == VAR_REF) && (s->variant == FUNCTION_NAME)) { 
                if(isResultVar(s))
	          (yyvsp[(1) - (5)].ll_node)->variant = ARRAY_REF;
                else
                  (yyvsp[(1) - (5)].ll_node)->variant = FUNC_CALL;
              }
	      if (((yyvsp[(1) - (5)].ll_node)->variant == VAR_REF) && (s->variant == PROGRAM_NAME)) {
                 errstr("The name '%s' is invalid in this context",s->ident,285);
                 (yyvsp[(1) - (5)].ll_node)->variant = FUNC_CALL;
              }
              /* l = $1; */
	      num_triplets = is_array_section_ref((yyvsp[(4) - (5)].ll_node));
	      switch ((yyvsp[(1) - (5)].ll_node)->variant)
              {
	      case TYPE_REF:
                   (yyvsp[(1) - (5)].ll_node)->variant = STRUCTURE_CONSTRUCTOR;                  
                   (yyvsp[(1) - (5)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (5)].ll_node);
                   (yyval.ll_node) = (yyvsp[(1) - (5)].ll_node);
                   (yyval.ll_node)->type =  lookup_type(s->parent); 
	          /* $$ = make_llnd(fi, STRUCTURE_CONSTRUCTOR, $1, $4, SMNULL);
	           $$->type = $1->type;*//*18.02.03*/
	           break;
	      case INTERFACE_REF:
	       /*  sym = resolve_overloading(s, $4);
	           if (sym != SMNULL)
	  	   {
	              l = make_llnd(fi, FUNC_CALL, $4, LLNULL, sym);
	              l->type = sym->type;
	              $$ = $1; $$->variant = OVERLOADED_CALL;
	              $$->entry.Template.ll_ptr1 = l;
	              $$->type = sym->type;
	           }
	           else {
	             errstr("can't resolve call %s", s->ident,310);
	           }
	           break;
                 */ /*podd 01.02.03*/

                   (yyvsp[(1) - (5)].ll_node)->variant = FUNC_CALL;

	      case FUNC_CALL:
                   (yyvsp[(1) - (5)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (5)].ll_node);
                   (yyval.ll_node) = (yyvsp[(1) - (5)].ll_node);
                   if(s->type) 
                     (yyval.ll_node)->type = s->type;
                   else
                     (yyval.ll_node)->type = global_default;
	           /*late_bind_if_needed($$);*/ /*podd 02.02.23*/
	           break;
	      case DEREF_OP:
              case ARRAY_REF:
	           /* array element */
	           if (num_triplets == 0) {
                       if ((yyvsp[(4) - (5)].ll_node) == LLNULL) {
                           s = (yyvsp[(1) - (5)].ll_node)->entry.Template.symbol = make_function(s->parent, TYNULL, LOCAL);
                           s->entry.func_decl.num_output = 1;
                           (yyvsp[(1) - (5)].ll_node)->variant = FUNC_CALL;
                           (yyval.ll_node) = (yyvsp[(1) - (5)].ll_node);
                       } else if ((yyvsp[(1) - (5)].ll_node)->type->variant == T_STRING) {
                           PTR_LLND temp = (yyvsp[(4) - (5)].ll_node);
                           int num_input = 0;

                           while (temp) {
                             ++num_input;
                             temp = temp->entry.Template.ll_ptr2;
                           }
                           (yyvsp[(1) - (5)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (5)].ll_node);
                           s = (yyvsp[(1) - (5)].ll_node)->entry.Template.symbol = make_function(s->parent, TYNULL, LOCAL);
                           s->entry.func_decl.num_output = 1;
                           s->entry.func_decl.num_input = num_input;
                           (yyvsp[(1) - (5)].ll_node)->variant = FUNC_CALL;
                           (yyval.ll_node) = (yyvsp[(1) - (5)].ll_node);
                       } else {
       	                   (yyvsp[(1) - (5)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (5)].ll_node);
	                   (yyval.ll_node) = (yyvsp[(1) - (5)].ll_node);
                           (yyval.ll_node)->type = (yyvsp[(1) - (5)].ll_node)->type->entry.ar_decl.base_type;
                       }
                   }
                   /* substring */
	           else if ((num_triplets == 1) && 
                            ((yyvsp[(1) - (5)].ll_node)->type->variant == T_STRING)) {
    	           /*
                     $1->entry.Template.ll_ptr1 = $4;
	             $$ = $1; $$->type = global_string;
                   */
	                  (yyval.ll_node) = make_llnd(fi, 
			  ARRAY_OP, LLNULL, LLNULL, SMNULL);
    	                  (yyval.ll_node)->entry.Template.ll_ptr1 = (yyvsp[(1) - (5)].ll_node);
       	                  (yyval.ll_node)->entry.Template.ll_ptr2 = (yyvsp[(4) - (5)].ll_node)->entry.Template.ll_ptr1;
	                  (yyval.ll_node)->type = global_string;
                   }           
                   /* array section */
                   else {
    	             (yyvsp[(1) - (5)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (5)].ll_node);
	             (yyval.ll_node) = (yyvsp[(1) - (5)].ll_node); tp = make_type(fi, T_ARRAY);     /**18.03.17*/
                     tp->entry.ar_decl.base_type = (yyvsp[(1) - (5)].ll_node)->type->entry.ar_decl.base_type; /**18.03.17 $1->type */
	             tp->entry.ar_decl.num_dimensions = num_triplets;
	             (yyval.ll_node)->type = tp;
                   }
	           break;
	      default:
                    if((yyvsp[(1) - (5)].ll_node)->entry.Template.symbol)
                      errstr("Can't subscript %s",(yyvsp[(1) - (5)].ll_node)->entry.Template.symbol->ident, 44);
                    else
	              err("Can't subscript",44);
             }
             /*if ($$->variant == T_POINTER) {
	        l = $$;
	        $$ = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	        $$->type = l->type->entry.Template.base_type;
	     }
              */  /*11.02.03*/

	     endioctl(); 
           ;}
    break;

  case 386:
#line 3177 "gram1.y"
    { int num_triplets;
	      PTR_SYMB s;
	      PTR_LLND l;

	      s = (yyvsp[(1) - (6)].ll_node)->entry.Template.symbol;
/*              if ($1->type->variant == T_POINTER) {
	         l = $1;
	         $1 = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	         $1->type = l->type->entry.Template.base_type;
	      } */
	      if (((yyvsp[(1) - (6)].ll_node)->type->variant != T_ARRAY) ||
                  ((yyvsp[(1) - (6)].ll_node)->type->entry.ar_decl.base_type->variant != T_STRING)) {
	         errstr("Can't take substring of %s", s->ident, 45);
              }
              else {
  	        num_triplets = is_array_section_ref((yyvsp[(4) - (6)].ll_node));
	           /* array element */
                if (num_triplets == 0) {
                   (yyvsp[(1) - (6)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (6)].ll_node);
                  /* $1->entry.Template.ll_ptr2 = $6;*/
	          /* $$ = $1;*/
                   l=(yyvsp[(1) - (6)].ll_node);
                   /*$$->type = $1->type->entry.ar_decl.base_type;*/
                   l->type = global_string;  /**18.03.17* $1->type->entry.ar_decl.base_type;*/
                }
                /* array section */
                else {
    	           (yyvsp[(1) - (6)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (6)].ll_node);
    	           /*$1->entry.Template.ll_ptr2 = $6;
	           $$ = $1; $$->type = make_type(fi, T_ARRAY);
                   $$->type->entry.ar_decl.base_type = $1->type;
	           $$->type->entry.ar_decl.num_dimensions = num_triplets;
                  */
                   l = (yyvsp[(1) - (6)].ll_node); l->type = make_type(fi, T_ARRAY);
                   l->type->entry.ar_decl.base_type = global_string;   /**18.03.17* $1->type*/
	           l->type->entry.ar_decl.num_dimensions = num_triplets;
               }
                (yyval.ll_node) = make_llnd(fi, ARRAY_OP, l, (yyvsp[(6) - (6)].ll_node), SMNULL);
	        (yyval.ll_node)->type = l->type;
              
              /* if ($$->variant == T_POINTER) {
	          l = $$;
	          $$ = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	          $$->type = l->type->entry.Template.base_type;
	       }
               */  /*11.02.03*/
             }
             endioctl();
          ;}
    break;

  case 387:
#line 3227 "gram1.y"
    {  int num_triplets;
	      PTR_LLND l,l1,l2;
              PTR_TYPE tp;

         /*   if ($1->variant == T_POINTER) {
	         l = $1;
	         $1 = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	         $1->type = l->type->entry.Template.base_type;
	      } */

              num_triplets = is_array_section_ref((yyvsp[(3) - (4)].ll_node));
              (yyval.ll_node) = (yyvsp[(1) - (4)].ll_node);
              l2 = (yyvsp[(1) - (4)].ll_node)->entry.Template.ll_ptr2;  
              l1 = (yyvsp[(1) - (4)].ll_node)->entry.Template.ll_ptr1;                
              if(l2 && l2->type->variant == T_STRING)/*substring*/
                if(num_triplets == 1){
	           l = make_llnd(fi, ARRAY_OP, LLNULL, LLNULL, SMNULL);
    	           l->entry.Template.ll_ptr1 = l2;
       	           l->entry.Template.ll_ptr2 = (yyvsp[(3) - (4)].ll_node)->entry.Template.ll_ptr1;
	           l->type = global_string; 
                   (yyval.ll_node)->entry.Template.ll_ptr2 = l;                                          
                } else
                   err("Can't subscript",44);
              else if (l2 && l2->type->variant == T_ARRAY) {
                 if(num_triplets > 0) { /*array section*/
                   tp = make_type(fi,T_ARRAY);
                   tp->entry.ar_decl.base_type = (yyvsp[(1) - (4)].ll_node)->type->entry.ar_decl.base_type;
                   tp->entry.ar_decl.num_dimensions = num_triplets;
                   (yyval.ll_node)->type = tp;
                   l2->entry.Template.ll_ptr1 = (yyvsp[(3) - (4)].ll_node);
                   l2->type = (yyval.ll_node)->type;   
                  }                 
                 else {  /*array element*/
                   l2->type = l2->type->entry.ar_decl.base_type;
                   l2->entry.Template.ll_ptr1 = (yyvsp[(3) - (4)].ll_node);   
                   if(l1->type->variant != T_ARRAY)  
                     (yyval.ll_node)->type = l2->type;
                 }
              } else 
                   {err("Can't subscript",44); /*fprintf(stderr,"%d  %d",$1->variant,l2);*/}
                   /*errstr("Can't subscript %s",l2->entry.Template.symbol->ident,441);*/
         ;}
    break;

  case 388:
#line 3271 "gram1.y"
    { int num_triplets;
	      PTR_LLND l,q;

          /*     if ($1->variant == T_POINTER) {
	         l = $1;
	         $1 = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	         $1->type = l->type->entry.Template.base_type;
	      } */

              (yyval.ll_node) = (yyvsp[(1) - (5)].ll_node);
	      if (((yyvsp[(1) - (5)].ll_node)->type->variant != T_ARRAY) &&
                  ((yyvsp[(1) - (5)].ll_node)->type->entry.ar_decl.base_type->variant != T_STRING)) {
	         err("Can't take substring",45);
              }
              else {
  	        num_triplets = is_array_section_ref((yyvsp[(3) - (5)].ll_node));
                l = (yyvsp[(1) - (5)].ll_node)->entry.Template.ll_ptr2;
                if(l) {
                /* array element */
	        if (num_triplets == 0) {
                   l->entry.Template.ll_ptr1 = (yyvsp[(3) - (5)].ll_node);       	           
                   l->type = global_string;
                }
                /* array section */
                else {	
    	             l->entry.Template.ll_ptr1 = (yyvsp[(3) - (5)].ll_node);
	             l->type = make_type(fi, T_ARRAY);
                     l->type->entry.ar_decl.base_type = global_string;
	             l->type->entry.ar_decl.num_dimensions = num_triplets;
                }
	        q = make_llnd(fi, ARRAY_OP, l, (yyvsp[(5) - (5)].ll_node), SMNULL);
	        q->type = l->type;
                (yyval.ll_node)->entry.Template.ll_ptr2 = q;
                if((yyvsp[(1) - (5)].ll_node)->entry.Template.ll_ptr1->type->variant != T_ARRAY)  
                     (yyval.ll_node)->type = q->type;
               }
             }
          ;}
    break;

  case 389:
#line 3313 "gram1.y"
    { PTR_TYPE t;
	      PTR_SYMB  field;
	    /*  PTR_BFND at_scope;*/
              PTR_LLND l;


/*              if ($1->variant == T_POINTER) {
	         l = $1;
	         $1 = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	         $1->type = l->type->entry.Template.base_type;
	      } */

	      t = (yyvsp[(1) - (3)].ll_node)->type; 
	      
	      if (( ( ((yyvsp[(1) - (3)].ll_node)->variant == VAR_REF) 
	          ||  ((yyvsp[(1) - (3)].ll_node)->variant == CONST_REF) 
                  ||  ((yyvsp[(1) - (3)].ll_node)->variant == ARRAY_REF)
                  ||  ((yyvsp[(1) - (3)].ll_node)->variant == RECORD_REF)) && (t->variant == T_DERIVED_TYPE)) 
	          ||((((yyvsp[(1) - (3)].ll_node)->variant == ARRAY_REF) || ((yyvsp[(1) - (3)].ll_node)->variant == RECORD_REF)) && (t->variant == T_ARRAY) &&
                      (t = t->entry.ar_decl.base_type) && (t->variant == T_DERIVED_TYPE))) 
                {
                 t->name = lookup_type_symbol(t->name);
	         if ((field = component(t->name, yytext))) {                   
	            l =  make_llnd(fi, VAR_REF, LLNULL, LLNULL, field);
                    l->type = field->type;
                    if(field->type->variant == T_ARRAY || field->type->variant == T_STRING)
                      l->variant = ARRAY_REF; 
                    (yyval.ll_node) = make_llnd(fi, RECORD_REF, (yyvsp[(1) - (3)].ll_node), l, SMNULL);
                    if((yyvsp[(1) - (3)].ll_node)->type->variant != T_ARRAY)
                       (yyval.ll_node)->type = field->type;
                    else {
                       (yyval.ll_node)->type = make_type(fi,T_ARRAY);
                       if(field->type->variant != T_ARRAY) 
	                 (yyval.ll_node)->type->entry.ar_decl.base_type = field->type;
                       else
                         (yyval.ll_node)->type->entry.ar_decl.base_type = field->type->entry.ar_decl.base_type;
	               (yyval.ll_node)->type->entry.ar_decl.num_dimensions = t->entry.ar_decl.num_dimensions;
                       }
                 }
                  else  
                    errstr("Illegal component  %s", yytext,311);
              }                     
               else 
                    errstr("Can't take component  %s", yytext,311);
             ;}
    break;

  case 390:
#line 3371 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 391:
#line 3373 "gram1.y"
    {(yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 392:
#line 3375 "gram1.y"
    {  int num_triplets;
               PTR_TYPE tp;
              /* PTR_LLND l;*/
	      if ((yyvsp[(1) - (5)].ll_node)->type->variant == T_ARRAY)
              {
  	         num_triplets = is_array_section_ref((yyvsp[(4) - (5)].ll_node));
	         /* array element */
	         if (num_triplets == 0) {
       	            (yyvsp[(1) - (5)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (5)].ll_node);
       	            (yyval.ll_node) = (yyvsp[(1) - (5)].ll_node);
                    (yyval.ll_node)->type = (yyvsp[(1) - (5)].ll_node)->type->entry.ar_decl.base_type;
                 }
                 /* substring */
	       /*  else if ((num_triplets == 1) && 
                          ($1->type->variant == T_STRING)) {
    	                  $1->entry.Template.ll_ptr1 = $4;
	                  $$ = $1; $$->type = global_string;
                 }   */ /*podd*/        
                 /* array section */
                 else {
    	             (yyvsp[(1) - (5)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (5)].ll_node);
	             (yyval.ll_node) = (yyvsp[(1) - (5)].ll_node); tp = make_type(fi, T_ARRAY);
                     tp->entry.ar_decl.base_type = (yyvsp[(1) - (5)].ll_node)->type->entry.ar_decl.base_type;  /**18.03.17* $1->type */
	             tp->entry.ar_decl.num_dimensions = num_triplets;
                     (yyval.ll_node)->type = tp;
                 }
             } 
             else err("can't subscript",44);

            /* if ($$->variant == T_POINTER) {
	        l = $$;
	        $$ = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	        $$->type = l->type->entry.Template.base_type;
	     }
             */  /*11.02.03*/

            endioctl();
           ;}
    break;

  case 393:
#line 3415 "gram1.y"
    {  int num_triplets;
	      PTR_LLND l,l1,l2;

         /*   if ($1->variant == T_POINTER) {
	         l = $1;
	         $1 = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	         $1->type = l->type->entry.Template.base_type;
	      } */

              num_triplets = is_array_section_ref((yyvsp[(3) - (4)].ll_node));
              (yyval.ll_node) = (yyvsp[(1) - (4)].ll_node);
              l2 = (yyvsp[(1) - (4)].ll_node)->entry.Template.ll_ptr2;  
              l1 = (yyvsp[(1) - (4)].ll_node)->entry.Template.ll_ptr1;                
              if(l2 && l2->type->variant == T_STRING)/*substring*/
                if(num_triplets == 1){
	           l = make_llnd(fi, ARRAY_OP, LLNULL, LLNULL, SMNULL);
    	           l->entry.Template.ll_ptr1 = l2;
       	           l->entry.Template.ll_ptr2 = (yyvsp[(3) - (4)].ll_node)->entry.Template.ll_ptr1;
	           l->type = global_string; 
                   (yyval.ll_node)->entry.Template.ll_ptr2 = l;                                          
                } else
                   err("Can't subscript",44);
              else if (l2 && l2->type->variant == T_ARRAY) {
                 if(num_triplets > 0) { /*array section*/
                   (yyval.ll_node)->type = make_type(fi,T_ARRAY);
                   (yyval.ll_node)->type->entry.ar_decl.base_type = l2->type->entry.ar_decl.base_type;
                   (yyval.ll_node)->type->entry.ar_decl.num_dimensions = num_triplets;
                   l2->entry.Template.ll_ptr1 = (yyvsp[(3) - (4)].ll_node);
                   l2->type = (yyval.ll_node)->type;   
                  }                 
                 else {  /*array element*/
                   l2->type = l2->type->entry.ar_decl.base_type;
                   l2->entry.Template.ll_ptr1 = (yyvsp[(3) - (4)].ll_node);   
                   if(l1->type->variant != T_ARRAY)  
                     (yyval.ll_node)->type = l2->type;
                 }
              } else 
                   err("Can't subscript",44);
         ;}
    break;

  case 394:
#line 3457 "gram1.y"
    { 
	      if ((yyvsp[(1) - (2)].ll_node)->type->variant == T_STRING) {
                 (yyvsp[(1) - (2)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(2) - (2)].ll_node);
                 (yyval.ll_node) = (yyvsp[(1) - (2)].ll_node); (yyval.ll_node)->type = global_string;
              }
              else errstr("can't subscript of %s", (yyvsp[(1) - (2)].ll_node)->entry.Template.symbol->ident,44);
            ;}
    break;

  case 395:
#line 3467 "gram1.y"
    { (yyval.ll_node) = LLNULL; ;}
    break;

  case 396:
#line 3469 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 397:
#line 3473 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, DDOT, (yyvsp[(2) - (5)].ll_node), (yyvsp[(4) - (5)].ll_node), SMNULL); ;}
    break;

  case 398:
#line 3477 "gram1.y"
    { (yyval.ll_node) = LLNULL; ;}
    break;

  case 399:
#line 3479 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 400:
#line 3483 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 401:
#line 3485 "gram1.y"
    { PTR_TYPE t;
               t = make_type_node((yyvsp[(1) - (3)].ll_node)->type, (yyvsp[(3) - (3)].ll_node));
               (yyval.ll_node) = (yyvsp[(1) - (3)].ll_node);
               (yyval.ll_node)->type = t;
             ;}
    break;

  case 402:
#line 3491 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 403:
#line 3493 "gram1.y"
    { PTR_TYPE t;
               t = make_type_node((yyvsp[(1) - (3)].ll_node)->type, (yyvsp[(3) - (3)].ll_node));
               (yyval.ll_node) = (yyvsp[(1) - (3)].ll_node);
               (yyval.ll_node)->type = t;
             ;}
    break;

  case 404:
#line 3499 "gram1.y"
    {
              if ((yyvsp[(2) - (2)].ll_node) != LLNULL)
              {
		 (yyval.ll_node) = make_llnd(fi, ARRAY_OP, (yyvsp[(1) - (2)].ll_node), (yyvsp[(2) - (2)].ll_node), SMNULL); 
                 (yyval.ll_node)->type = global_string;
              }
	      else 
                 (yyval.ll_node) = (yyvsp[(1) - (2)].ll_node);
             ;}
    break;

  case 405:
#line 3512 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,BOOL_VAL, LLNULL, LLNULL, SMNULL);
	      (yyval.ll_node)->entry.bval = 1;
	      (yyval.ll_node)->type = global_bool;
	    ;}
    break;

  case 406:
#line 3518 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,BOOL_VAL, LLNULL, LLNULL, SMNULL);
	      (yyval.ll_node)->entry.bval = 0;
	      (yyval.ll_node)->type = global_bool;
	    ;}
    break;

  case 407:
#line 3525 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,FLOAT_VAL, LLNULL, LLNULL, SMNULL);
	      (yyval.ll_node)->entry.string_val = copys(yytext);
	      (yyval.ll_node)->type = global_float;
	    ;}
    break;

  case 408:
#line 3531 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,DOUBLE_VAL, LLNULL, LLNULL, SMNULL);
	      (yyval.ll_node)->entry.string_val = copys(yytext);
	      (yyval.ll_node)->type = global_double;
	    ;}
    break;

  case 409:
#line 3539 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,INT_VAL, LLNULL, LLNULL, SMNULL);
	      (yyval.ll_node)->entry.ival = atoi(yytext);
	      (yyval.ll_node)->type = global_int;
	    ;}
    break;

  case 410:
#line 3547 "gram1.y"
    { PTR_TYPE t;
	      PTR_LLND p,q;
	      (yyval.ll_node) = make_llnd(fi,STRING_VAL, LLNULL, LLNULL, SMNULL);
	      (yyval.ll_node)->entry.string_val = copys(yytext);
              if(yyquote=='\"') 
	        t = global_string_2;
              else
	        t = global_string;

	      p = make_llnd(fi,INT_VAL, LLNULL, LLNULL, SMNULL);
	      p->entry.ival = yyleng;
	      p->type = global_int;
              q = make_llnd(fi, LEN_OP, p, LLNULL, SMNULL); 
              (yyval.ll_node)->type = make_type_node(t, q);
	    ;}
    break;

  case 411:
#line 3563 "gram1.y"
    { PTR_TYPE t;
	      (yyval.ll_node) = make_llnd(fi,STRING_VAL, LLNULL, LLNULL, SMNULL);
	      (yyval.ll_node)->entry.string_val = copys(yytext);
              if(yyquote=='\"') 
	        t = global_string_2;
              else
	        t = global_string;
	      (yyval.ll_node)->type = make_type_node(t, (yyvsp[(1) - (3)].ll_node));
            ;}
    break;

  case 412:
#line 3573 "gram1.y"
    { PTR_TYPE t;
	      (yyval.ll_node) = make_llnd(fi,STRING_VAL, LLNULL, LLNULL, SMNULL);
	      (yyval.ll_node)->entry.string_val = copys(yytext);
              if(yyquote=='\"') 
	        t = global_string_2;
              else
	        t = global_string;
	      (yyval.ll_node)->type = make_type_node(t, (yyvsp[(1) - (3)].ll_node));
            ;}
    break;

  case 413:
#line 3586 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,COMPLEX_VAL, (yyvsp[(2) - (5)].ll_node), (yyvsp[(4) - (5)].ll_node), SMNULL);
	      (yyval.ll_node)->type = global_complex;
	    ;}
    break;

  case 414:
#line 3593 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 415:
#line 3595 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 416:
#line 3618 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT,(yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),SMNULL); ;}
    break;

  case 417:
#line 3620 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT,(yyvsp[(1) - (2)].ll_node),LLNULL,SMNULL); ;}
    break;

  case 418:
#line 3622 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT,make_llnd(fi,DDOT,(yyvsp[(1) - (5)].ll_node),(yyvsp[(3) - (5)].ll_node),SMNULL),(yyvsp[(5) - (5)].ll_node),SMNULL); ;}
    break;

  case 419:
#line 3624 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT,make_llnd(fi,DDOT,(yyvsp[(1) - (4)].ll_node),LLNULL,SMNULL),(yyvsp[(4) - (4)].ll_node),SMNULL); ;}
    break;

  case 420:
#line 3626 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT, make_llnd(fi,DDOT,LLNULL,(yyvsp[(2) - (4)].ll_node),SMNULL),(yyvsp[(4) - (4)].ll_node),SMNULL); ;}
    break;

  case 421:
#line 3628 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT,make_llnd(fi,DDOT,LLNULL,LLNULL,SMNULL),(yyvsp[(3) - (3)].ll_node),SMNULL); ;}
    break;

  case 422:
#line 3630 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT,LLNULL,(yyvsp[(2) - (2)].ll_node),SMNULL); ;}
    break;

  case 423:
#line 3632 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT,LLNULL,LLNULL,SMNULL); ;}
    break;

  case 424:
#line 3635 "gram1.y"
    {in_vec=YES;;}
    break;

  case 425:
#line 3635 "gram1.y"
    {in_vec=NO;;}
    break;

  case 426:
#line 3636 "gram1.y"
    { PTR_TYPE array_type;
             (yyval.ll_node) = make_llnd (fi,CONSTRUCTOR_REF,(yyvsp[(4) - (6)].ll_node),LLNULL,SMNULL); 
             /*$$->type = $2->type;*/ /*28.02.03*/
             array_type = make_type(fi, T_ARRAY);
	     array_type->entry.ar_decl.num_dimensions = 1;
             if((yyvsp[(4) - (6)].ll_node)->type->variant == T_ARRAY)
	       array_type->entry.ar_decl.base_type = (yyvsp[(4) - (6)].ll_node)->type->entry.ar_decl.base_type;
             else
               array_type->entry.ar_decl.base_type = (yyvsp[(4) - (6)].ll_node)->type;
             (yyval.ll_node)->type = array_type;
           ;}
    break;

  case 427:
#line 3650 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 428:
#line 3652 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 429:
#line 3675 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 430:
#line 3677 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (4)].ll_node), (yyvsp[(4) - (4)].ll_node), EXPR_LIST); endioctl(); ;}
    break;

  case 431:
#line 3679 "gram1.y"
    { stat_alloc = make_llnd(fi, SPEC_PAIR, (yyvsp[(4) - (5)].ll_node), (yyvsp[(5) - (5)].ll_node), SMNULL);
                  endioctl();
                ;}
    break;

  case 432:
#line 3695 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 433:
#line 3697 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (4)].ll_node), (yyvsp[(4) - (4)].ll_node), EXPR_LIST); endioctl(); ;}
    break;

  case 434:
#line 3699 "gram1.y"
    { stat_alloc = make_llnd(fi, SPEC_PAIR, (yyvsp[(4) - (5)].ll_node), (yyvsp[(5) - (5)].ll_node), SMNULL);
             endioctl();
           ;}
    break;

  case 435:
#line 3712 "gram1.y"
    {stat_alloc = LLNULL;;}
    break;

  case 436:
#line 3716 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 437:
#line 3718 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); ;}
    break;

  case 438:
#line 3726 "gram1.y"
    { (yyval.bf_node) = (yyvsp[(1) - (1)].bf_node); ;}
    break;

  case 439:
#line 3728 "gram1.y"
    { (yyval.bf_node) = (yyvsp[(1) - (1)].bf_node); ;}
    break;

  case 440:
#line 3730 "gram1.y"
    { (yyval.bf_node) = (yyvsp[(1) - (1)].bf_node); ;}
    break;

  case 441:
#line 3732 "gram1.y"
    {
              (yyval.bf_node) = (yyvsp[(2) - (2)].bf_node);
              (yyval.bf_node)->entry.Template.ll_ptr3 = (yyvsp[(1) - (2)].ll_node);
            ;}
    break;

  case 442:
#line 3786 "gram1.y"
    { PTR_BFND biff;

	      (yyval.bf_node) = get_bfnd(fi,CONTROL_END, SMNULL, LLNULL, LLNULL, LLNULL); 
	      bind(); 
	      biff = cur_scope();
	      if ((biff->variant == FUNC_HEDR) || (biff->variant == PROC_HEDR)
		  || (biff->variant == PROS_HEDR) 
	          || (biff->variant == PROG_HEDR)
                  || (biff->variant == BLOCK_DATA)) {
                if(biff->control_parent == global_bfnd) position = IN_OUTSIDE;
		else if(!is_interface_stat(biff->control_parent)) position++;
              } else if  (biff->variant == MODULE_STMT)
                position = IN_OUTSIDE;
	      else err("Unexpected END statement read", 52);
             /* FB ADDED set the control parent so the empty function unparse right*/
              if ((yyval.bf_node))
                (yyval.bf_node)->control_parent = biff;
              delete_beyond_scope_level(pred_bfnd);
            ;}
    break;

  case 443:
#line 3808 "gram1.y"
    {
              make_extend((yyvsp[(3) - (3)].symbol));
              (yyval.bf_node) = BFNULL; 
              /* delete_beyond_scope_level(pred_bfnd); */
             ;}
    break;

  case 444:
#line 3821 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,CONTROL_END, SMNULL, LLNULL, LLNULL, LLNULL); 
	    bind(); 
	    delete_beyond_scope_level(pred_bfnd);
	    position = IN_OUTSIDE;
          ;}
    break;

  case 445:
#line 3830 "gram1.y"
    { (yyval.bf_node) = (yyvsp[(1) - (1)].bf_node); ;}
    break;

  case 446:
#line 3833 "gram1.y"
    {
              (yyval.bf_node) = (yyvsp[(2) - (2)].bf_node);
              (yyval.bf_node)->entry.Template.ll_ptr3 = (yyvsp[(1) - (2)].ll_node);
            ;}
    break;

  case 447:
#line 3883 "gram1.y"
    { thiswasbranch = NO;
              (yyvsp[(1) - (2)].bf_node)->variant = LOGIF_NODE;
              (yyval.bf_node) = make_logif((yyvsp[(1) - (2)].bf_node), (yyvsp[(2) - (2)].bf_node));
	      set_blobs((yyvsp[(1) - (2)].bf_node), pred_bfnd, SAME_GROUP);
	    ;}
    break;

  case 448:
#line 3889 "gram1.y"
    {
              (yyval.bf_node) = (yyvsp[(1) - (2)].bf_node);
	      set_blobs((yyval.bf_node), pred_bfnd, NEW_GROUP1); 
            ;}
    break;

  case 449:
#line 3894 "gram1.y"
    {
              (yyval.bf_node) = (yyvsp[(2) - (3)].bf_node);
              (yyval.bf_node)->entry.Template.ll_ptr3 = (yyvsp[(1) - (3)].ll_node);
	      set_blobs((yyval.bf_node), pred_bfnd, NEW_GROUP1); 
            ;}
    break;

  case 450:
#line 3912 "gram1.y"
    { make_elseif((yyvsp[(4) - (7)].ll_node),(yyvsp[(7) - (7)].symbol)); lastwasbranch = NO; (yyval.bf_node) = BFNULL;;}
    break;

  case 451:
#line 3914 "gram1.y"
    { make_else((yyvsp[(3) - (3)].symbol)); lastwasbranch = NO; (yyval.bf_node) = BFNULL; ;}
    break;

  case 452:
#line 3916 "gram1.y"
    { make_endif((yyvsp[(3) - (3)].symbol)); (yyval.bf_node) = BFNULL; ;}
    break;

  case 453:
#line 3918 "gram1.y"
    { (yyval.bf_node) = (yyvsp[(1) - (1)].bf_node); ;}
    break;

  case 454:
#line 3920 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, CONTAINS_STMT, SMNULL, LLNULL, LLNULL, LLNULL); ;}
    break;

  case 455:
#line 3923 "gram1.y"
    { thiswasbranch = NO;
              (yyvsp[(1) - (2)].bf_node)->variant = FORALL_STAT;
              (yyval.bf_node) = make_logif((yyvsp[(1) - (2)].bf_node), (yyvsp[(2) - (2)].bf_node));
	      set_blobs((yyvsp[(1) - (2)].bf_node), pred_bfnd, SAME_GROUP);
	    ;}
    break;

  case 456:
#line 3929 "gram1.y"
    { (yyval.bf_node) = (yyvsp[(1) - (1)].bf_node); ;}
    break;

  case 457:
#line 3931 "gram1.y"
    { (yyval.bf_node) = (yyvsp[(2) - (2)].bf_node); (yyval.bf_node)->entry.Template.ll_ptr3 = (yyvsp[(1) - (2)].ll_node);;}
    break;

  case 458:
#line 3933 "gram1.y"
    { make_endforall((yyvsp[(3) - (3)].symbol)); (yyval.bf_node) = BFNULL; ;}
    break;

  case 459:
#line 3936 "gram1.y"
    { (yyval.bf_node) = (yyvsp[(1) - (1)].bf_node); ;}
    break;

  case 460:
#line 3938 "gram1.y"
    { (yyval.bf_node) = (yyvsp[(1) - (1)].bf_node); ;}
    break;

  case 461:
#line 3940 "gram1.y"
    { (yyval.bf_node) = (yyvsp[(1) - (1)].bf_node); ;}
    break;

  case 462:
#line 3967 "gram1.y"
    { 	     
	     /*  if($5 && $5->labdefined)
		 execerr("no backward DO loops", (char *)NULL); */
	       (yyval.bf_node) = make_do(WHILE_NODE, LBNULL, SMNULL, (yyvsp[(4) - (5)].ll_node), LLNULL, LLNULL);
	       /*$$->entry.Template.ll_ptr3 = $1;*/	     
           ;}
    break;

  case 463:
#line 3976 "gram1.y"
    {
               if( (yyvsp[(4) - (7)].label) && (yyvsp[(4) - (7)].label)->labdefined)
		  err("No backward DO loops", 46);
	        (yyval.bf_node) = make_do(WHILE_NODE, (yyvsp[(4) - (7)].label), SMNULL, (yyvsp[(7) - (7)].ll_node), LLNULL, LLNULL);            
	    ;}
    break;

  case 464:
#line 3984 "gram1.y"
    { (yyval.ll_node) = LLNULL; ;}
    break;

  case 465:
#line 3986 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(5) - (6)].ll_node);;}
    break;

  case 466:
#line 3988 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(3) - (4)].ll_node);;}
    break;

  case 467:
#line 3993 "gram1.y"
    {  
               if( (yyvsp[(4) - (11)].label) && (yyvsp[(4) - (11)].label)->labdefined)
		  err("No backward DO loops", 46);
	        (yyval.bf_node) = make_do(FOR_NODE, (yyvsp[(4) - (11)].label), (yyvsp[(7) - (11)].symbol), (yyvsp[(9) - (11)].ll_node), (yyvsp[(11) - (11)].ll_node), LLNULL);            
	    ;}
    break;

  case 468:
#line 4000 "gram1.y"
    {
               if( (yyvsp[(4) - (13)].label) && (yyvsp[(4) - (13)].label)->labdefined)
		  err("No backward DO loops", 46);
	        (yyval.bf_node) = make_do(FOR_NODE, (yyvsp[(4) - (13)].label), (yyvsp[(7) - (13)].symbol), (yyvsp[(9) - (13)].ll_node), (yyvsp[(11) - (13)].ll_node), (yyvsp[(13) - (13)].ll_node));            
	    ;}
    break;

  case 469:
#line 4008 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, CASE_NODE, (yyvsp[(4) - (4)].symbol), (yyvsp[(3) - (4)].ll_node), LLNULL, LLNULL); ;}
    break;

  case 470:
#line 4010 "gram1.y"
    { /*PTR_LLND p;*/
	     /* p = make_llnd(fi, DEFAULT, LLNULL, LLNULL, SMNULL); */
	      (yyval.bf_node) = get_bfnd(fi, DEFAULT_NODE, (yyvsp[(3) - (3)].symbol), LLNULL, LLNULL, LLNULL); ;}
    break;

  case 471:
#line 4014 "gram1.y"
    { make_endselect((yyvsp[(3) - (3)].symbol)); (yyval.bf_node) = BFNULL; ;}
    break;

  case 472:
#line 4017 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, SWITCH_NODE, SMNULL, (yyvsp[(6) - (7)].ll_node), LLNULL, LLNULL) ; ;}
    break;

  case 473:
#line 4019 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, SWITCH_NODE, SMNULL, (yyvsp[(7) - (8)].ll_node), LLNULL, (yyvsp[(1) - (8)].ll_node)) ; ;}
    break;

  case 474:
#line 4023 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(2) - (3)].ll_node); ;}
    break;

  case 475:
#line 4029 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 476:
#line 4031 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, DDOT, (yyvsp[(1) - (2)].ll_node), LLNULL, SMNULL); ;}
    break;

  case 477:
#line 4033 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, DDOT, LLNULL, (yyvsp[(2) - (2)].ll_node), SMNULL); ;}
    break;

  case 478:
#line 4035 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, DDOT, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), SMNULL); ;}
    break;

  case 479:
#line 4039 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, EXPR_LIST, (yyvsp[(1) - (1)].ll_node), LLNULL, SMNULL); ;}
    break;

  case 480:
#line 4041 "gram1.y"
    { PTR_LLND p;
	      
	      p = make_llnd(fi, EXPR_LIST, (yyvsp[(3) - (3)].ll_node), LLNULL, SMNULL);
	      add_to_lowLevelList(p, (yyvsp[(1) - (3)].ll_node));
	    ;}
    break;

  case 481:
#line 4049 "gram1.y"
    { (yyval.symbol) = SMNULL; ;}
    break;

  case 482:
#line 4051 "gram1.y"
    { (yyval.symbol) = make_local_entity((yyvsp[(1) - (1)].hash_entry), CONSTRUCT_NAME, global_default,
                                     LOCAL); ;}
    break;

  case 483:
#line 4057 "gram1.y"
    {(yyval.hash_entry) = HSNULL;;}
    break;

  case 484:
#line 4059 "gram1.y"
    { (yyval.hash_entry) = (yyvsp[(1) - (1)].hash_entry);;}
    break;

  case 485:
#line 4063 "gram1.y"
    {(yyval.hash_entry) = look_up_sym(yytext);;}
    break;

  case 486:
#line 4067 "gram1.y"
    { PTR_SYMB s;
	             s = make_local_entity( (yyvsp[(1) - (2)].hash_entry), CONSTRUCT_NAME, global_default, LOCAL);             
                    (yyval.ll_node) = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
                   ;}
    break;

  case 487:
#line 4088 "gram1.y"
    { (yyval.bf_node) = make_if((yyvsp[(4) - (5)].ll_node)); ;}
    break;

  case 488:
#line 4091 "gram1.y"
    { (yyval.bf_node) = make_forall((yyvsp[(4) - (6)].ll_node),(yyvsp[(5) - (6)].ll_node)); ;}
    break;

  case 489:
#line 4095 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, EXPR_LIST, (yyvsp[(1) - (1)].ll_node), LLNULL, SMNULL); ;}
    break;

  case 490:
#line 4097 "gram1.y"
    { PTR_LLND p;	      
	      p = make_llnd(fi, EXPR_LIST, (yyvsp[(3) - (3)].ll_node), LLNULL, SMNULL);
	      add_to_lowLevelList(p, (yyvsp[(1) - (3)].ll_node));
	    ;}
    break;

  case 491:
#line 4104 "gram1.y"
    {(yyval.ll_node) = make_llnd(fi, FORALL_OP, (yyvsp[(3) - (3)].ll_node), LLNULL, (yyvsp[(1) - (3)].symbol)); ;}
    break;

  case 492:
#line 4108 "gram1.y"
    { (yyval.ll_node)=LLNULL;;}
    break;

  case 493:
#line 4110 "gram1.y"
    { (yyval.ll_node)=(yyvsp[(2) - (2)].ll_node);;}
    break;

  case 494:
#line 4121 "gram1.y"
    { PTR_SYMB  s;
              s = (yyvsp[(1) - (1)].hash_entry)->id_attr;
      	      if (!s || s->variant == DEFAULT)
              {
	         s = make_scalar((yyvsp[(1) - (1)].hash_entry), TYNULL, LOCAL);
	     	 s->decl = SOFT;
	      }
              (yyval.symbol) = s; 
	 ;}
    break;

  case 495:
#line 4134 "gram1.y"
    { PTR_SYMB s;
              PTR_LLND l;
              int vrnt;

            /*  s = make_scalar($1, TYNULL, LOCAL);*/ /*16.02.03*/
              s = (yyvsp[(1) - (5)].symbol);
	      if (s->variant != CONST_NAME) {
                if(in_vec) 
                   vrnt=SEQ;
                else
                   vrnt=DDOT;     
                l = make_llnd(fi, SEQ, make_llnd(fi, vrnt, (yyvsp[(3) - (5)].ll_node), (yyvsp[(5) - (5)].ll_node), SMNULL),
                              LLNULL, SMNULL);
		(yyval.ll_node) = make_llnd(fi,IOACCESS, LLNULL, l, s);
		do_name_err = NO;
	      }
	      else {
		err("Symbolic constant not allowed as DO variable", 47);
		do_name_err = YES;
	      }
	    ;}
    break;

  case 496:
#line 4157 "gram1.y"
    { PTR_SYMB s;
              PTR_LLND l;
              int vrnt;
              /*s = make_scalar($1, TYNULL, LOCAL);*/ /*16.02.03*/
              s = (yyvsp[(1) - (7)].symbol);
	      if( s->variant != CONST_NAME ) {
                if(in_vec) 
                   vrnt=SEQ;
                else
                   vrnt=DDOT;     
                l = make_llnd(fi, SEQ, make_llnd(fi, vrnt, (yyvsp[(3) - (7)].ll_node), (yyvsp[(5) - (7)].ll_node), SMNULL), (yyvsp[(7) - (7)].ll_node),
                              SMNULL);
		(yyval.ll_node) = make_llnd(fi,IOACCESS, LLNULL, l, s);
		do_name_err = NO;
	      }
	      else {
		err("Symbolic constant not allowed as DO variable", 47);
		do_name_err = YES;
	      }
	    ;}
    break;

  case 497:
#line 4180 "gram1.y"
    { (yyval.label) = LBNULL; ;}
    break;

  case 498:
#line 4182 "gram1.y"
    {
	       (yyval.label)  = make_label_node(fi,convci(yyleng, yytext));
	       (yyval.label)->scope = cur_scope();
	    ;}
    break;

  case 499:
#line 4189 "gram1.y"
    { make_endwhere((yyvsp[(3) - (3)].symbol)); (yyval.bf_node) = BFNULL; ;}
    break;

  case 500:
#line 4191 "gram1.y"
    { make_elsewhere((yyvsp[(3) - (3)].symbol)); lastwasbranch = NO; (yyval.bf_node) = BFNULL; ;}
    break;

  case 501:
#line 4193 "gram1.y"
    { make_elsewhere_mask((yyvsp[(4) - (6)].ll_node),(yyvsp[(6) - (6)].symbol)); lastwasbranch = NO; (yyval.bf_node) = BFNULL; ;}
    break;

  case 502:
#line 4195 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, WHERE_BLOCK_STMT, SMNULL, (yyvsp[(4) - (5)].ll_node), LLNULL, LLNULL); ;}
    break;

  case 503:
#line 4197 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, WHERE_BLOCK_STMT, SMNULL, (yyvsp[(5) - (6)].ll_node), LLNULL, (yyvsp[(1) - (6)].ll_node)); ;}
    break;

  case 504:
#line 4202 "gram1.y"
    { PTR_LLND p, r;
             PTR_SYMB s1, s2 = SMNULL, s3, arg_list;
	     PTR_HASH hash_entry;

	   /*  if (just_look_up_sym("=") != HSNULL) {
	        p = intrinsic_op_node("=", EQUAL, $2, $4);
   	        $$ = get_bfnd(fi, OVERLOADED_ASSIGN_STAT, SMNULL, p, $2, $4);
             }	      
             else */ if ((yyvsp[(2) - (4)].ll_node)->variant == FUNC_CALL) {
                if(parstate==INEXEC){
                  	  err("Declaration among executables", 30);
                 /*   $$=BFNULL;*/
 	         (yyval.bf_node) = get_bfnd(fi,STMTFN_STAT, SMNULL, (yyvsp[(2) - (4)].ll_node), LLNULL, LLNULL);
                } 
                else {	         
  	         (yyvsp[(2) - (4)].ll_node)->variant = STMTFN_DECL;
		 /* $2->entry.Template.ll_ptr2 = $4; */
                 if( (yyvsp[(2) - (4)].ll_node)->entry.Template.ll_ptr1) {
		   r = (yyvsp[(2) - (4)].ll_node)->entry.Template.ll_ptr1->entry.Template.ll_ptr1;
                   if(r->variant != VAR_REF && r->variant != ARRAY_REF){
                     err("A dummy argument of a statement function must be a scalar identifier", 333);
                     s1 = SMNULL;
                   }
                   else                       
		     s1 = r ->entry.Template.symbol;
                 } else
                   s1 = SMNULL;
		 if (s1)
	            s1->scope = cur_scope();
 	         (yyval.bf_node) = get_bfnd(fi,STMTFN_STAT, SMNULL, (yyvsp[(2) - (4)].ll_node), LLNULL, LLNULL);
	         add_scope_level((yyval.bf_node), NO);
                 arg_list = SMNULL;
		 if (s1) 
                 {
	            /*arg_list = SMNULL;*/
                    p = (yyvsp[(2) - (4)].ll_node)->entry.Template.ll_ptr1;
                    while (p != LLNULL)
                    {
		    /*   if (p->entry.Template.ll_ptr1->variant != VAR_REF) {
			  errstr("cftn.gram:1: illegal statement function %s.", $2->entry.Template.symbol->ident);
			  break;
		       } 
                    */
                       r = p->entry.Template.ll_ptr1;
                       if(r->variant != VAR_REF && r->variant != ARRAY_REF){
                         err("A dummy argument of a statement function must be a scalar identifier", 333);
                         break;
                       }
	               hash_entry = look_up_sym(r->entry.Template.symbol->parent->ident);
	               s3 = make_scalar(hash_entry, s1->type, IO);
                       replace_symbol_in_expr(s3,(yyvsp[(4) - (4)].ll_node));
	               if (arg_list == SMNULL) 
                          s2 = arg_list = s3;
             	       else 
                       {
                          s2->id_list = s3;
                          s2 = s3;
                       }
                       p = p->entry.Template.ll_ptr2;
                    }
                 }
  		    (yyvsp[(2) - (4)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (4)].ll_node);
		    install_param_list((yyvsp[(2) - (4)].ll_node)->entry.Template.symbol,
				       arg_list, LLNULL, FUNCTION_NAME);
	            delete_beyond_scope_level((yyval.bf_node));
		 
		/* else
		    errstr("cftn.gram: Illegal statement function declaration %s.", $2->entry.Template.symbol->ident); */
               }
	     }
	     else {
		(yyval.bf_node) = get_bfnd(fi,ASSIGN_STAT,SMNULL, (yyvsp[(2) - (4)].ll_node), (yyvsp[(4) - (4)].ll_node), LLNULL);
                 parstate = INEXEC;
             }
	  ;}
    break;

  case 505:
#line 4278 "gram1.y"
    { /*PTR_SYMB s;*/
	
	      /*s = make_scalar($2, TYNULL, LOCAL);*/
  	      (yyval.bf_node) = get_bfnd(fi, POINTER_ASSIGN_STAT, SMNULL, (yyvsp[(3) - (5)].ll_node), (yyvsp[(5) - (5)].ll_node), LLNULL);
	    ;}
    break;

  case 506:
#line 4290 "gram1.y"
    { PTR_SYMB p;

	      p = make_scalar((yyvsp[(5) - (5)].hash_entry), TYNULL, LOCAL);
	      p->variant = LABEL_VAR;
  	      (yyval.bf_node) = get_bfnd(fi,ASSLAB_STAT, p, (yyvsp[(3) - (5)].ll_node),LLNULL,LLNULL);
            ;}
    break;

  case 507:
#line 4297 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,CONT_STAT,SMNULL,LLNULL,LLNULL,LLNULL); ;}
    break;

  case 509:
#line 4300 "gram1.y"
    { inioctl = NO; ;}
    break;

  case 510:
#line 4302 "gram1.y"
    { PTR_LLND	p;

	      p = make_llnd(fi,EXPR_LIST, (yyvsp[(10) - (10)].ll_node), LLNULL, SMNULL);
	      p = make_llnd(fi,EXPR_LIST, (yyvsp[(8) - (10)].ll_node), p, SMNULL);
	      (yyval.bf_node)= get_bfnd(fi,ARITHIF_NODE, SMNULL, (yyvsp[(4) - (10)].ll_node),
			    make_llnd(fi,EXPR_LIST, (yyvsp[(6) - (10)].ll_node), p, SMNULL), LLNULL);
	      thiswasbranch = YES;
            ;}
    break;

  case 511:
#line 4311 "gram1.y"
    {
	      (yyval.bf_node) = subroutine_call((yyvsp[(1) - (1)].symbol), LLNULL, LLNULL, PLAIN);
/*	      match_parameters($1, LLNULL);
	      $$= get_bfnd(fi,PROC_STAT, $1, LLNULL, LLNULL, LLNULL);
*/	      endioctl(); 
            ;}
    break;

  case 512:
#line 4318 "gram1.y"
    {
	      (yyval.bf_node) = subroutine_call((yyvsp[(1) - (3)].symbol), LLNULL, LLNULL, PLAIN);
/*	      match_parameters($1, LLNULL);
	      $$= get_bfnd(fi,PROC_STAT,$1,LLNULL,LLNULL,LLNULL);
*/	      endioctl(); 
	    ;}
    break;

  case 513:
#line 4325 "gram1.y"
    {
	      (yyval.bf_node) = subroutine_call((yyvsp[(1) - (4)].symbol), (yyvsp[(3) - (4)].ll_node), LLNULL, PLAIN);
/*	      match_parameters($1, $3);
	      $$= get_bfnd(fi,PROC_STAT,$1,$3,LLNULL,LLNULL);
*/	      endioctl(); 
	    ;}
    break;

  case 514:
#line 4333 "gram1.y"
    {
	      (yyval.bf_node) = get_bfnd(fi,RETURN_STAT,SMNULL,(yyvsp[(3) - (3)].ll_node),LLNULL,LLNULL);
	      thiswasbranch = YES;
	    ;}
    break;

  case 515:
#line 4338 "gram1.y"
    {
	      (yyval.bf_node) = get_bfnd(fi,(yyvsp[(1) - (3)].token),SMNULL,(yyvsp[(3) - (3)].ll_node),LLNULL,LLNULL);
	      thiswasbranch = ((yyvsp[(1) - (3)].token) == STOP_STAT);
	    ;}
    break;

  case 516:
#line 4343 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, CYCLE_STMT, (yyvsp[(3) - (3)].symbol), LLNULL, LLNULL, LLNULL); ;}
    break;

  case 517:
#line 4346 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, EXIT_STMT, (yyvsp[(3) - (3)].symbol), LLNULL, LLNULL, LLNULL); ;}
    break;

  case 518:
#line 4349 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, ALLOCATE_STMT,  SMNULL, (yyvsp[(5) - (6)].ll_node), stat_alloc, LLNULL); ;}
    break;

  case 519:
#line 4352 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, DEALLOCATE_STMT, SMNULL, (yyvsp[(5) - (6)].ll_node), stat_alloc , LLNULL); ;}
    break;

  case 520:
#line 4355 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, NULLIFY_STMT, SMNULL, (yyvsp[(4) - (5)].ll_node), LLNULL, LLNULL); ;}
    break;

  case 521:
#line 4358 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi, WHERE_NODE, SMNULL, (yyvsp[(4) - (8)].ll_node), (yyvsp[(6) - (8)].ll_node), (yyvsp[(8) - (8)].ll_node)); ;}
    break;

  case 522:
#line 4376 "gram1.y"
    {(yyval.ll_node) = LLNULL;;}
    break;

  case 523:
#line 4380 "gram1.y"
    {
	      (yyval.bf_node)=get_bfnd(fi,GOTO_NODE,SMNULL,LLNULL,LLNULL,(PTR_LLND)(yyvsp[(3) - (3)].ll_node));
	      thiswasbranch = YES;
	    ;}
    break;

  case 524:
#line 4385 "gram1.y"
    { PTR_SYMB p;

	      if((yyvsp[(3) - (3)].hash_entry)->id_attr)
		p = (yyvsp[(3) - (3)].hash_entry)->id_attr;
	      else {
	        p = make_scalar((yyvsp[(3) - (3)].hash_entry), TYNULL, LOCAL);
		p->variant = LABEL_VAR;
	      }

	      if(p->variant == LABEL_VAR) {
		  (yyval.bf_node) = get_bfnd(fi,ASSGOTO_NODE,p,LLNULL,LLNULL,LLNULL);
		  thiswasbranch = YES;
	      }
	      else {
		  err("Must go to assigned variable", 48);
		  (yyval.bf_node) = BFNULL;
	      }
	    ;}
    break;

  case 525:
#line 4404 "gram1.y"
    { PTR_SYMB p;

	      if((yyvsp[(3) - (7)].hash_entry)->id_attr)
		p = (yyvsp[(3) - (7)].hash_entry)->id_attr;
	      else {
	        p = make_scalar((yyvsp[(3) - (7)].hash_entry), TYNULL, LOCAL);
		p->variant = LABEL_VAR;
	      }

	      if (p->variant == LABEL_VAR) {
		 (yyval.bf_node) = get_bfnd(fi,ASSGOTO_NODE,p,(yyvsp[(6) - (7)].ll_node),LLNULL,LLNULL);
		 thiswasbranch = YES;
	      }
	      else {
		err("Must go to assigned variable",48);
		(yyval.bf_node) = BFNULL;
	      }
	    ;}
    break;

  case 526:
#line 4423 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,COMGOTO_NODE, SMNULL, (yyvsp[(4) - (7)].ll_node), (yyvsp[(7) - (7)].ll_node), LLNULL); ;}
    break;

  case 529:
#line 4431 "gram1.y"
    { (yyval.symbol) = make_procedure((yyvsp[(3) - (4)].hash_entry), LOCAL); ;}
    break;

  case 530:
#line 4435 "gram1.y"
    { 
              (yyval.ll_node) = set_ll_list((yyvsp[(2) - (2)].ll_node), LLNULL, EXPR_LIST);
              endioctl();
            ;}
    break;

  case 531:
#line 4440 "gram1.y"
    { 
               (yyval.ll_node) = set_ll_list((yyvsp[(1) - (4)].ll_node), (yyvsp[(4) - (4)].ll_node), EXPR_LIST);
               endioctl();
            ;}
    break;

  case 532:
#line 4447 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 533:
#line 4449 "gram1.y"
    { (yyval.ll_node)  = make_llnd(fi, KEYWORD_ARG, (yyvsp[(1) - (2)].ll_node), (yyvsp[(2) - (2)].ll_node), SMNULL); ;}
    break;

  case 534:
#line 4451 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,LABEL_ARG,(yyvsp[(2) - (2)].ll_node),LLNULL,SMNULL); ;}
    break;

  case 535:
#line 4454 "gram1.y"
    { (yyval.token) = PAUSE_NODE; ;}
    break;

  case 536:
#line 4455 "gram1.y"
    { (yyval.token) = STOP_STAT; ;}
    break;

  case 537:
#line 4466 "gram1.y"
    { if(parstate == OUTSIDE)
		{ PTR_BFND p;

		  p = get_bfnd(fi,PROG_HEDR, make_program(look_up_sym("_MAIN")), LLNULL, LLNULL, LLNULL);
		  set_blobs(p, global_bfnd, NEW_GROUP1);
		  add_scope_level(p, NO);
		  position = IN_PROC; 
		}
		if(parstate < INDATA) enddcl();
		parstate = INEXEC;
		yystno = 0;
	      ;}
    break;

  case 538:
#line 4481 "gram1.y"
    { intonly = YES; ;}
    break;

  case 539:
#line 4485 "gram1.y"
    { intonly = NO; ;}
    break;

  case 540:
#line 4493 "gram1.y"
    { (yyvsp[(1) - (2)].bf_node)->entry.Template.ll_ptr2 = (yyvsp[(2) - (2)].ll_node);
		  (yyval.bf_node) = (yyvsp[(1) - (2)].bf_node); ;}
    break;

  case 541:
#line 4496 "gram1.y"
    { PTR_LLND p, q = LLNULL;

		  q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  q->entry.string_val = (char *)"unit";
		  q->type = global_string;
		  p = make_llnd(fi, SPEC_PAIR, q, (yyvsp[(2) - (2)].ll_node), SMNULL);
		  (yyvsp[(1) - (2)].bf_node)->entry.Template.ll_ptr2 = p;
		  endioctl();
		  (yyval.bf_node) = (yyvsp[(1) - (2)].bf_node); ;}
    break;

  case 542:
#line 4506 "gram1.y"
    { PTR_LLND p, q, r;

		  p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  p->entry.string_val = (char *)"*";
		  p->type = global_string;
		  q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  q->entry.string_val = (char *)"unit";
		  q->type = global_string;
		  r = make_llnd(fi, SPEC_PAIR, p, q, SMNULL);
		  (yyvsp[(1) - (2)].bf_node)->entry.Template.ll_ptr2 = r;
		  endioctl();
		  (yyval.bf_node) = (yyvsp[(1) - (2)].bf_node); ;}
    break;

  case 543:
#line 4519 "gram1.y"
    { PTR_LLND p, q, r;

		  p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  p->entry.string_val = (char *)"**";
		  p->type = global_string;
		  q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  q->entry.string_val = (char *)"unit";
		  q->type = global_string;
		  r = make_llnd(fi, SPEC_PAIR, p, q, SMNULL);
		  (yyvsp[(1) - (2)].bf_node)->entry.Template.ll_ptr2 = r;
		  endioctl();
		  (yyval.bf_node) = (yyvsp[(1) - (2)].bf_node); ;}
    break;

  case 544:
#line 4532 "gram1.y"
    { (yyvsp[(1) - (2)].bf_node)->entry.Template.ll_ptr2 = (yyvsp[(2) - (2)].ll_node);
		  (yyval.bf_node) = (yyvsp[(1) - (2)].bf_node); ;}
    break;

  case 545:
#line 4535 "gram1.y"
    { (yyval.bf_node) = (yyvsp[(1) - (1)].bf_node); ;}
    break;

  case 546:
#line 4537 "gram1.y"
    { (yyvsp[(1) - (2)].bf_node)->entry.Template.ll_ptr2 = (yyvsp[(2) - (2)].ll_node);
		  (yyval.bf_node) = (yyvsp[(1) - (2)].bf_node); ;}
    break;

  case 547:
#line 4540 "gram1.y"
    { (yyvsp[(1) - (2)].bf_node)->entry.Template.ll_ptr2 = (yyvsp[(2) - (2)].ll_node);
		  (yyval.bf_node) = (yyvsp[(1) - (2)].bf_node); ;}
    break;

  case 548:
#line 4543 "gram1.y"
    { (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr2 = (yyvsp[(2) - (3)].ll_node);
		  (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1 = (yyvsp[(3) - (3)].ll_node);
		  (yyval.bf_node) = (yyvsp[(1) - (3)].bf_node); ;}
    break;

  case 549:
#line 4547 "gram1.y"
    { (yyvsp[(1) - (4)].bf_node)->entry.Template.ll_ptr2 = (yyvsp[(2) - (4)].ll_node);
		  (yyvsp[(1) - (4)].bf_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (4)].ll_node);
		  (yyval.bf_node) = (yyvsp[(1) - (4)].bf_node); ;}
    break;

  case 550:
#line 4556 "gram1.y"
    { (yyvsp[(1) - (2)].bf_node)->entry.Template.ll_ptr2 = (yyvsp[(2) - (2)].ll_node);
		  (yyval.bf_node) = (yyvsp[(1) - (2)].bf_node); ;}
    break;

  case 551:
#line 4559 "gram1.y"
    { (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr2 = (yyvsp[(2) - (3)].ll_node);
		  (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1 = (yyvsp[(3) - (3)].ll_node);
		  (yyval.bf_node) = (yyvsp[(1) - (3)].bf_node); ;}
    break;

  case 552:
#line 4563 "gram1.y"
    { (yyval.bf_node) = (yyvsp[(1) - (1)].bf_node); ;}
    break;

  case 553:
#line 4565 "gram1.y"
    { (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1 = (yyvsp[(3) - (3)].ll_node);
		  (yyval.bf_node) = (yyvsp[(1) - (3)].bf_node); ;}
    break;

  case 554:
#line 4571 "gram1.y"
    { (yyval.bf_node) = (yyvsp[(1) - (3)].bf_node); ;}
    break;

  case 555:
#line 4575 "gram1.y"
    {(yyval.bf_node) = get_bfnd(fi, BACKSPACE_STAT, SMNULL, LLNULL, LLNULL, LLNULL);;}
    break;

  case 556:
#line 4577 "gram1.y"
    {(yyval.bf_node) = get_bfnd(fi, REWIND_STAT, SMNULL, LLNULL, LLNULL, LLNULL);;}
    break;

  case 557:
#line 4579 "gram1.y"
    {(yyval.bf_node) = get_bfnd(fi, ENDFILE_STAT, SMNULL, LLNULL, LLNULL, LLNULL);;}
    break;

  case 558:
#line 4586 "gram1.y"
    { (yyval.bf_node) = (yyvsp[(1) - (3)].bf_node); ;}
    break;

  case 559:
#line 4590 "gram1.y"
    {(yyval.bf_node) = get_bfnd(fi, OPEN_STAT, SMNULL, LLNULL, LLNULL, LLNULL);;}
    break;

  case 560:
#line 4592 "gram1.y"
    {(yyval.bf_node) = get_bfnd(fi, CLOSE_STAT, SMNULL, LLNULL, LLNULL, LLNULL);;}
    break;

  case 561:
#line 4596 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi, INQUIRE_STAT, SMNULL, LLNULL, (yyvsp[(4) - (4)].ll_node), LLNULL);;}
    break;

  case 562:
#line 4598 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi, INQUIRE_STAT, SMNULL, (yyvsp[(5) - (5)].ll_node), (yyvsp[(4) - (5)].ll_node), LLNULL);;}
    break;

  case 563:
#line 4602 "gram1.y"
    { PTR_LLND p;
		  PTR_LLND q = LLNULL;

		  if ((yyvsp[(1) - (1)].ll_node)->variant == INT_VAL)
 	          {
		        PTR_LABEL r;

			r = make_label_node(fi, (long) (yyvsp[(1) - (1)].ll_node)->entry.ival);
			r->scope = cur_scope();
			p = make_llnd_label(fi, LABEL_REF, r);
		  }
		  else p = (yyvsp[(1) - (1)].ll_node); 
		  q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  q->entry.string_val = (char *)"fmt";
		  q->type = global_string;
		  (yyval.ll_node) = make_llnd(fi, SPEC_PAIR, q, p, SMNULL);
		  endioctl();
		;}
    break;

  case 564:
#line 4621 "gram1.y"
    { PTR_LLND p;
		  PTR_LLND q;

		  p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  p->entry.string_val = (char *)"*";
		  p->type = global_string;
		  q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  q->entry.string_val = (char *)"fmt";
		  q->type = global_string;
		  (yyval.ll_node) = make_llnd(fi, SPEC_PAIR, q, p, SMNULL);
		  endioctl();
		;}
    break;

  case 565:
#line 4637 "gram1.y"
    { PTR_LLND p;

		  p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  p->entry.string_val = (char *)"unit";
		  p->type = global_string;
		  (yyval.ll_node) = make_llnd(fi, SPEC_PAIR, p, (yyvsp[(2) - (3)].ll_node), SMNULL);
		  endioctl();
		;}
    break;

  case 566:
#line 4648 "gram1.y"
    { 
		  (yyval.ll_node) = (yyvsp[(2) - (3)].ll_node);
		  endioctl();
		 ;}
    break;

  case 567:
#line 4655 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); endioctl();;}
    break;

  case 568:
#line 4657 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (4)].ll_node), (yyvsp[(4) - (4)].ll_node), EXPR_LIST); endioctl();;}
    break;

  case 569:
#line 4661 "gram1.y"
    { PTR_LLND p;
		  PTR_LLND q;
 
		  nioctl++;
		  if ((nioctl == 2) && ((yyvsp[(1) - (1)].ll_node)->variant == INT_VAL))
 	          {
		        PTR_LABEL r;

			r = make_label_node(fi, (long) (yyvsp[(1) - (1)].ll_node)->entry.ival);
			r->scope = cur_scope();
			p = make_llnd_label(fi, LABEL_REF, r);
		  }
		  else p = (yyvsp[(1) - (1)].ll_node); 
		  q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  if (nioctl == 1)
		        q->entry.string_val = (char *)"unit"; 
		  else {
                     if(((yyvsp[(1) - (1)].ll_node)->variant == VAR_REF) && (yyvsp[(1) - (1)].ll_node)->entry.Template.symbol->variant == NAMELIST_NAME)
                       q->entry.string_val = (char *)"nml";
                     else
                       q->entry.string_val = (char *)"fmt";
                  }
		  q->type = global_string;
		  (yyval.ll_node) = make_llnd(fi, SPEC_PAIR, q, p, SMNULL);
		;}
    break;

  case 570:
#line 4687 "gram1.y"
    { PTR_LLND p;
		  PTR_LLND q;

		  nioctl++;
		  p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  p->entry.string_val = (char *)"*";
		  p->type = global_string;
		  q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  if (nioctl == 1)
		        q->entry.string_val = (char *)"unit"; 
		  else  q->entry.string_val = (char *)"fmt";
		  q->type = global_string;
		  (yyval.ll_node) = make_llnd(fi, SPEC_PAIR, q, p, SMNULL);
		;}
    break;

  case 571:
#line 4702 "gram1.y"
    { PTR_LLND p;
		  PTR_LLND q;

		  nioctl++;
		  p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  p->entry.string_val = (char *)"**";
		  p->type = global_string;
		  q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  if (nioctl == 1)
		        q->entry.string_val = (char *)"unit"; 
		  else  q->entry.string_val = (char *)"fmt";
		  q->type = global_string;
		  (yyval.ll_node) = make_llnd(fi, SPEC_PAIR, q, p, SMNULL);
		;}
    break;

  case 572:
#line 4717 "gram1.y"
    { 
		  PTR_LLND p;
		  char *q;

		  q = (yyvsp[(1) - (2)].ll_node)->entry.string_val;
  		  if ((strcmp(q, "end") == 0) || (strcmp(q, "err") == 0) || (strcmp(q, "eor") == 0) || ((strcmp(q,"fmt") == 0) && ((yyvsp[(2) - (2)].ll_node)->variant == INT_VAL)))
 	          {
		        PTR_LABEL r;

			r = make_label_node(fi, (long) (yyvsp[(2) - (2)].ll_node)->entry.ival);
			r->scope = cur_scope();
			p = make_llnd_label(fi, LABEL_REF, r);
		  }
		  else p = (yyvsp[(2) - (2)].ll_node);

		  (yyval.ll_node) = make_llnd(fi, SPEC_PAIR, (yyvsp[(1) - (2)].ll_node), p, SMNULL); ;}
    break;

  case 573:
#line 4734 "gram1.y"
    { PTR_LLND p;
                  
		  p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  p->entry.string_val = (char *)"*";
		  p->type = global_string;
		  (yyval.ll_node) = make_llnd(fi, SPEC_PAIR, (yyvsp[(1) - (2)].ll_node), p, SMNULL);
		;}
    break;

  case 574:
#line 4742 "gram1.y"
    { PTR_LLND p;
		  p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  p->entry.string_val = (char *)"*";
		  p->type = global_string;
		  (yyval.ll_node) = make_llnd(fi, SPEC_PAIR, (yyvsp[(1) - (2)].ll_node), p, SMNULL);
		;}
    break;

  case 575:
#line 4751 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  (yyval.ll_node)->entry.string_val = copys(yytext);
		  (yyval.ll_node)->type = global_string;
	        ;}
    break;

  case 576:
#line 4759 "gram1.y"
    {(yyval.bf_node) = get_bfnd(fi, READ_STAT, SMNULL, LLNULL, LLNULL, LLNULL);;}
    break;

  case 577:
#line 4764 "gram1.y"
    {(yyval.bf_node) = get_bfnd(fi, WRITE_STAT, SMNULL, LLNULL, LLNULL, LLNULL);;}
    break;

  case 578:
#line 4769 "gram1.y"
    {
	    PTR_LLND p, q, l;

	    if ((yyvsp[(3) - (4)].ll_node)->variant == INT_VAL)
		{
		        PTR_LABEL r;

			r = make_label_node(fi, (long) (yyvsp[(3) - (4)].ll_node)->entry.ival);
			r->scope = cur_scope();
			p = make_llnd_label(fi, LABEL_REF, r);
		}
	    else p = (yyvsp[(3) - (4)].ll_node);
	    
            q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
	    q->entry.string_val = (char *)"fmt";
            q->type = global_string;
            l = make_llnd(fi, SPEC_PAIR, q, p, SMNULL);

            (yyval.bf_node) = get_bfnd(fi, PRINT_STAT, SMNULL, LLNULL, l, LLNULL);
	    endioctl();
	   ;}
    break;

  case 579:
#line 4791 "gram1.y"
    { PTR_LLND p, q, r;
		
	     p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
	     p->entry.string_val = (char *)"*";
	     p->type = global_string;
	     q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
	     q->entry.string_val = (char *)"fmt";
             q->type = global_string;
             r = make_llnd(fi, SPEC_PAIR, q, p, SMNULL);
	     (yyval.bf_node) = get_bfnd(fi, PRINT_STAT, SMNULL, LLNULL, r, LLNULL);
	     endioctl();
           ;}
    break;

  case 580:
#line 4807 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST);;}
    break;

  case 581:
#line 4809 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST);;}
    break;

  case 582:
#line 4813 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 583:
#line 4815 "gram1.y"
    {
		  (yyvsp[(4) - (5)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(2) - (5)].ll_node);
		  (yyval.ll_node) = (yyvsp[(4) - (5)].ll_node);
		;}
    break;

  case 584:
#line 4822 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST);  (yyval.ll_node)->type = (yyvsp[(1) - (1)].ll_node)->type;;}
    break;

  case 585:
#line 4824 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 586:
#line 4826 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 587:
#line 4830 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); (yyval.ll_node)->type = (yyvsp[(1) - (3)].ll_node)->type;;}
    break;

  case 588:
#line 4832 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); (yyval.ll_node)->type = (yyvsp[(1) - (3)].ll_node)->type;;}
    break;

  case 589:
#line 4834 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); (yyval.ll_node)->type = (yyvsp[(1) - (3)].ll_node)->type;;}
    break;

  case 590:
#line 4836 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); (yyval.ll_node)->type = (yyvsp[(1) - (3)].ll_node)->type;;}
    break;

  case 591:
#line 4838 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); (yyval.ll_node)->type = (yyvsp[(1) - (3)].ll_node)->type;;}
    break;

  case 592:
#line 4840 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); (yyval.ll_node)->type = (yyvsp[(1) - (3)].ll_node)->type;;}
    break;

  case 593:
#line 4844 "gram1.y"
    { (yyval.ll_node) =  set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST);
	          (yyval.ll_node)->type = global_complex; ;}
    break;

  case 594:
#line 4847 "gram1.y"
    { (yyval.ll_node) =  set_ll_list((yyvsp[(2) - (3)].ll_node), LLNULL, EXPR_LIST);
                  (yyval.ll_node)->type = (yyvsp[(2) - (3)].ll_node)->type; ;}
    break;

  case 595:
#line 4850 "gram1.y"
    {
		  (yyvsp[(4) - (5)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(2) - (5)].ll_node);
		  (yyval.ll_node) =  set_ll_list((yyvsp[(4) - (5)].ll_node), LLNULL, EXPR_LIST);
                  (yyval.ll_node)->type = (yyvsp[(2) - (5)].ll_node)->type; 
		;}
    break;

  case 596:
#line 4856 "gram1.y"
    {
		  (yyvsp[(4) - (5)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(2) - (5)].ll_node);
		  (yyval.ll_node) =  set_ll_list((yyvsp[(4) - (5)].ll_node), LLNULL, EXPR_LIST);
                  (yyval.ll_node)->type = (yyvsp[(2) - (5)].ll_node)->type; 
		;}
    break;

  case 597:
#line 4862 "gram1.y"
    {
		  (yyvsp[(4) - (5)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(2) - (5)].ll_node);
		  (yyval.ll_node) =  set_ll_list((yyvsp[(4) - (5)].ll_node), LLNULL, EXPR_LIST);
                  (yyval.ll_node)->type = (yyvsp[(2) - (5)].ll_node)->type; 
		;}
    break;

  case 598:
#line 4870 "gram1.y"
    { inioctl = YES; ;}
    break;

  case 599:
#line 4874 "gram1.y"
    { startioctl();;}
    break;

  case 600:
#line 4882 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 601:
#line 4884 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(2) - (3)].ll_node); ;}
    break;

  case 602:
#line 4888 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 603:
#line 4890 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 604:
#line 4892 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,(yyvsp[(2) - (3)].token), (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), SMNULL);
	      set_expr_type((yyval.ll_node));
	    ;}
    break;

  case 605:
#line 4897 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,MULT_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), SMNULL);
	      set_expr_type((yyval.ll_node));
	    ;}
    break;

  case 606:
#line 4902 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,DIV_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), SMNULL);
	      set_expr_type((yyval.ll_node));
	    ;}
    break;

  case 607:
#line 4907 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,EXP_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), SMNULL);
	      set_expr_type((yyval.ll_node));
	    ;}
    break;

  case 608:
#line 4912 "gram1.y"
    {
	      if((yyvsp[(1) - (2)].token) == SUBT_OP)
		{
		  (yyval.ll_node) = make_llnd(fi,SUBT_OP, (yyvsp[(2) - (2)].ll_node), LLNULL, SMNULL);
		  set_expr_type((yyval.ll_node));
		}
	      else	(yyval.ll_node) = (yyvsp[(2) - (2)].ll_node);
	    ;}
    break;

  case 609:
#line 4921 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,CONCAT_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), SMNULL);
	      set_expr_type((yyval.ll_node));
	    ;}
    break;

  case 610:
#line 4926 "gram1.y"
    { (yyval.ll_node) = LLNULL; ;}
    break;

  case 611:
#line 4931 "gram1.y"
    { comments = cur_comment = CMNULL; ;}
    break;

  case 612:
#line 4933 "gram1.y"
    { PTR_CMNT p;
	    p = make_comment(fi,*commentbuf, HALF);
	    if (cur_comment)
               cur_comment->next = p;
            else {
	       if ((pred_bfnd->control_parent->variant == LOGIF_NODE) ||(pred_bfnd->control_parent->variant == FORALL_STAT))

	           pred_bfnd->control_parent->entry.Template.cmnt_ptr = p;

	       else last_bfnd->entry.Template.cmnt_ptr = p;
            }
	    comments = cur_comment = CMNULL;
          ;}
    break;

  case 676:
#line 5016 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,HPF_TEMPLATE_STAT, SMNULL, (yyvsp[(3) - (3)].ll_node), LLNULL, LLNULL); ;}
    break;

  case 677:
#line 5018 "gram1.y"
    { PTR_SYMB s;
                if((yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr2)
                {
                  s = (yyvsp[(3) - (3)].ll_node)->entry.Template.ll_ptr1->entry.Template.symbol;
                  s->attr = s->attr | COMMON_BIT;
                }
	        add_to_lowLevelList((yyvsp[(3) - (3)].ll_node), (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);
	      ;}
    break;

  case 678:
#line 5029 "gram1.y"
    {PTR_SYMB s;
	      PTR_LLND q;
	    /* 27.06.18
	      if(! explicit_shape)   
                err("Explicit shape specification is required", 50);
	    */  
	      s = make_array((yyvsp[(1) - (2)].hash_entry), TYNULL, (yyvsp[(2) - (2)].ll_node), ndim, LOCAL);
              if(s->attr & TEMPLATE_BIT)
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
              if((s->attr & PROCESSORS_BIT) || (s->attr & TASK_BIT) || (s->attr & DVM_POINTER_BIT))
                errstr( "Inconsistent declaration of identifier  %s ", s->ident, 16);
              else
	        s->attr = s->attr | TEMPLATE_BIT;
              if((yyvsp[(2) - (2)].ll_node)) s->attr = s->attr | DIMENSION_BIT;  
	      q = make_llnd(fi,ARRAY_REF, (yyvsp[(2) - (2)].ll_node), LLNULL, s);
	      s->type->entry.ar_decl.ranges = (yyvsp[(2) - (2)].ll_node);
	      (yyval.ll_node) = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	     ;}
    break;

  case 679:
#line 5050 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_DYNAMIC_DIR, SMNULL, (yyvsp[(3) - (3)].ll_node), LLNULL, LLNULL);;}
    break;

  case 680:
#line 5054 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 681:
#line 5056 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 682:
#line 5060 "gram1.y"
    {  PTR_SYMB s;
	      s = make_array((yyvsp[(1) - (1)].hash_entry), TYNULL, LLNULL, 0, LOCAL);
              if(s->attr &  DYNAMIC_BIT)
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
              if((s->attr & PROCESSORS_BIT) || (s->attr & TASK_BIT) || (s->attr & HEAP_BIT)) 
                errstr("Inconsistent declaration of identifier  %s", s->ident, 16); 
              else
                s->attr = s->attr | DYNAMIC_BIT;        
	      (yyval.ll_node) = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);
	   ;}
    break;

  case 683:
#line 5073 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_INHERIT_DIR, SMNULL, (yyvsp[(3) - (3)].ll_node), LLNULL, LLNULL);;}
    break;

  case 684:
#line 5077 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 685:
#line 5079 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 686:
#line 5083 "gram1.y"
    {  PTR_SYMB s;
	      s = make_array((yyvsp[(1) - (1)].hash_entry), TYNULL, LLNULL, 0, LOCAL);
              if((s->attr & PROCESSORS_BIT) ||(s->attr & TASK_BIT)  || (s->attr & TEMPLATE_BIT) || (s->attr & ALIGN_BIT) || (s->attr & DISTRIBUTE_BIT)) 
                errstr("Inconsistent declaration of identifier  %s", s->ident, 16); 
              else
                s->attr = s->attr | INHERIT_BIT;        
	      (yyval.ll_node) = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);
	   ;}
    break;

  case 687:
#line 5094 "gram1.y"
    { PTR_LLND q;
             q = set_ll_list((yyvsp[(3) - (4)].ll_node),LLNULL,EXPR_LIST);
              /* (void)fprintf(stderr,"hpf.gram: shadow\n");*/ 
	     (yyval.bf_node) = get_bfnd(fi,DVM_SHADOW_DIR,SMNULL,q,(yyvsp[(4) - (4)].ll_node),LLNULL);
            ;}
    break;

  case 688:
#line 5105 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(2) - (3)].ll_node);;}
    break;

  case 689:
#line 5109 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 690:
#line 5111 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 691:
#line 5115 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 692:
#line 5117 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), SMNULL);;}
    break;

  case 693:
#line 5119 "gram1.y"
    {
            if(parstate!=INEXEC) 
               err("Illegal shadow width specification", 56);  
            (yyval.ll_node) = make_llnd(fi,SHADOW_NAMES_OP, (yyvsp[(3) - (4)].ll_node), LLNULL, SMNULL);
          ;}
    break;

  case 694:
#line 5134 "gram1.y"
    {  PTR_SYMB s;
	      s = make_array((yyvsp[(1) - (1)].hash_entry), TYNULL, LLNULL, 0, LOCAL);
              if(s->attr & SHADOW_BIT)
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
              if((s->attr & PROCESSORS_BIT) ||(s->attr & TASK_BIT)  || (s->attr & TEMPLATE_BIT) || (s->attr & HEAP_BIT)) 
                      errstr( "Inconsistent declaration of identifier %s", s->ident, 16); 
              else
        	      s->attr = s->attr | SHADOW_BIT;  
	      (yyval.ll_node) = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);
	   ;}
    break;

  case 695:
#line 5146 "gram1.y"
    { PTR_SYMB s;
	      PTR_LLND q, r;
	      if(! explicit_shape) {
              err("Explicit shape specification is required", 50);
		/* $$ = BFNULL;*/
	      }
	      s = make_array((yyvsp[(3) - (4)].hash_entry), TYNULL, (yyvsp[(4) - (4)].ll_node), ndim, LOCAL);
              if(s->attr &  PROCESSORS_BIT)
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
              if((s->attr & ALIGN_BIT) ||(s->attr & DISTRIBUTE_BIT) ||(s->attr & TEMPLATE_BIT) || (s->attr & DYNAMIC_BIT) ||(s->attr & SHADOW_BIT) || (s->attr & TASK_BIT) || (s->attr & DVM_POINTER_BIT) || (s->attr & INHERIT_BIT))
                errstr("Inconsistent declaration of identifier %s", s->ident, 16);
              else
	        s->attr = s->attr | PROCESSORS_BIT;
              if((yyvsp[(4) - (4)].ll_node)) s->attr = s->attr | DIMENSION_BIT;
	      q = make_llnd(fi,ARRAY_REF, (yyvsp[(4) - (4)].ll_node), LLNULL, s);
	      s->type->entry.ar_decl.ranges = (yyvsp[(4) - (4)].ll_node);
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      (yyval.bf_node) = get_bfnd(fi,HPF_PROCESSORS_STAT, SMNULL, r, LLNULL, LLNULL);
	    ;}
    break;

  case 696:
#line 5166 "gram1.y"
    { PTR_SYMB s;
	      PTR_LLND q, r;
	      if(! explicit_shape) {
              err("Explicit shape specification is required", 50);
		/* $$ = BFNULL;*/
	      }
	      s = make_array((yyvsp[(3) - (4)].hash_entry), TYNULL, (yyvsp[(4) - (4)].ll_node), ndim, LOCAL);
              if(s->attr &  PROCESSORS_BIT)
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
              if((s->attr & ALIGN_BIT) ||(s->attr & DISTRIBUTE_BIT) ||(s->attr & TEMPLATE_BIT) || (s->attr & DYNAMIC_BIT) ||(s->attr & SHADOW_BIT) || (s->attr & TASK_BIT) || (s->attr & DVM_POINTER_BIT) || (s->attr & INHERIT_BIT))
                errstr("Inconsistent declaration of identifier %s", s->ident, 16);
              else
	        s->attr = s->attr | PROCESSORS_BIT;
              if((yyvsp[(4) - (4)].ll_node)) s->attr = s->attr | DIMENSION_BIT;
	      q = make_llnd(fi,ARRAY_REF, (yyvsp[(4) - (4)].ll_node), LLNULL, s);
	      s->type->entry.ar_decl.ranges = (yyvsp[(4) - (4)].ll_node);
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      (yyval.bf_node) = get_bfnd(fi,HPF_PROCESSORS_STAT, SMNULL, r, LLNULL, LLNULL);
	    ;}
    break;

  case 697:
#line 5186 "gram1.y"
    {  PTR_SYMB s;
	      PTR_LLND q, r;
	      if(! explicit_shape) {
		err("Explicit shape specification is required", 50);
		/*$$ = BFNULL;*/
	      }
	      s = make_array((yyvsp[(3) - (4)].hash_entry), TYNULL, (yyvsp[(4) - (4)].ll_node), ndim, LOCAL);
              if(s->attr &  PROCESSORS_BIT)
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
              if((s->attr & ALIGN_BIT) ||(s->attr & DISTRIBUTE_BIT) ||(s->attr & TEMPLATE_BIT) || (s->attr & DYNAMIC_BIT) ||(s->attr & SHADOW_BIT) || (s->attr & TASK_BIT) || (s->attr &  DVM_POINTER_BIT) || (s->attr & INHERIT_BIT) )
                errstr("Inconsistent declaration of identifier  %s", s->ident, 16);
              else
	        s->attr = s->attr | PROCESSORS_BIT;
              if((yyvsp[(4) - (4)].ll_node)) s->attr = s->attr | DIMENSION_BIT;
	      q = make_llnd(fi,ARRAY_REF, (yyvsp[(4) - (4)].ll_node), LLNULL, s);
	      s->type->entry.ar_decl.ranges = (yyvsp[(4) - (4)].ll_node);
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      add_to_lowLevelList(r, (yyvsp[(1) - (4)].bf_node)->entry.Template.ll_ptr1);
	;}
    break;

  case 698:
#line 5208 "gram1.y"
    {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, (yyvsp[(3) - (3)].symbol));
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	           (yyval.bf_node) = get_bfnd(fi,DVM_INDIRECT_GROUP_DIR, SMNULL, r, LLNULL, LLNULL);
                ;}
    break;

  case 699:
#line 5214 "gram1.y"
    {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, (yyvsp[(3) - (3)].symbol));
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
                   add_to_lowLevelList(r, (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);
	           ;
                ;}
    break;

  case 700:
#line 5223 "gram1.y"
    {(yyval.symbol) = make_local_entity((yyvsp[(1) - (1)].hash_entry), REF_GROUP_NAME,global_default,LOCAL);
          if((yyval.symbol)->attr &  INDIRECT_BIT)
                errstr( "Multiple declaration of identifier  %s ", (yyval.symbol)->ident, 73);
           (yyval.symbol)->attr = (yyval.symbol)->attr | INDIRECT_BIT;
          ;}
    break;

  case 701:
#line 5231 "gram1.y"
    {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, (yyvsp[(3) - (3)].symbol));
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	           (yyval.bf_node) = get_bfnd(fi,DVM_REMOTE_GROUP_DIR, SMNULL, r, LLNULL, LLNULL);
                ;}
    break;

  case 702:
#line 5237 "gram1.y"
    {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, (yyvsp[(3) - (3)].symbol));
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
                   add_to_lowLevelList(r, (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);
                ;}
    break;

  case 703:
#line 5245 "gram1.y"
    {(yyval.symbol) = make_local_entity((yyvsp[(1) - (1)].hash_entry), REF_GROUP_NAME,global_default,LOCAL);
           if((yyval.symbol)->attr &  INDIRECT_BIT)
                errstr( "Inconsistent declaration of identifier  %s ", (yyval.symbol)->ident, 16);
          ;}
    break;

  case 704:
#line 5252 "gram1.y"
    {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, (yyvsp[(3) - (3)].symbol));
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	           (yyval.bf_node) = get_bfnd(fi,DVM_REDUCTION_GROUP_DIR, SMNULL, r, LLNULL, LLNULL);
                ;}
    break;

  case 705:
#line 5258 "gram1.y"
    {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, (yyvsp[(3) - (3)].symbol));
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
                   add_to_lowLevelList(r, (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);
	           ;
                ;}
    break;

  case 706:
#line 5267 "gram1.y"
    {(yyval.symbol) = make_local_entity((yyvsp[(1) - (1)].hash_entry), REDUCTION_GROUP_NAME,global_default,LOCAL);;}
    break;

  case 707:
#line 5271 "gram1.y"
    {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, (yyvsp[(3) - (3)].symbol));
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	           (yyval.bf_node) = get_bfnd(fi,DVM_CONSISTENT_GROUP_DIR, SMNULL, r, LLNULL, LLNULL);
                ;}
    break;

  case 708:
#line 5277 "gram1.y"
    {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, (yyvsp[(3) - (3)].symbol));
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
                   add_to_lowLevelList(r, (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);	           
                ;}
    break;

  case 709:
#line 5285 "gram1.y"
    {(yyval.symbol) = make_local_entity((yyvsp[(1) - (1)].hash_entry), CONSISTENT_GROUP_NAME,global_default,LOCAL);;}
    break;

  case 710:
#line 5299 "gram1.y"
    { PTR_SYMB s;
            if(parstate == INEXEC){
              if (!(s = (yyvsp[(2) - (3)].hash_entry)->id_attr))
              {
	         s = make_array((yyvsp[(2) - (3)].hash_entry), TYNULL, LLNULL, 0, LOCAL);
	     	 s->decl = SOFT;
	      } 
            } else
              s = make_array((yyvsp[(2) - (3)].hash_entry), TYNULL, LLNULL, 0, LOCAL);

              (yyval.ll_node) = make_llnd(fi,ARRAY_REF, (yyvsp[(3) - (3)].ll_node), LLNULL, s);
            ;}
    break;

  case 711:
#line 5312 "gram1.y"
    { (yyval.ll_node) = LLNULL; opt_kwd_ = NO;;}
    break;

  case 712:
#line 5318 "gram1.y"
    { PTR_LLND q;
             if(!(yyvsp[(4) - (5)].ll_node))
               err("Distribution format list is omitted", 51);
            /* if($6)
               err("NEW_VALUE specification in DISTRIBUTE directive");*/
             q = set_ll_list((yyvsp[(3) - (5)].ll_node),LLNULL,EXPR_LIST);
	     (yyval.bf_node) = get_bfnd(fi,DVM_DISTRIBUTE_DIR,SMNULL,q,(yyvsp[(4) - (5)].ll_node),(yyvsp[(5) - (5)].ll_node));
            ;}
    break;

  case 713:
#line 5334 "gram1.y"
    { PTR_LLND q;
                /*  if(!$4)
                  {err("Distribution format is omitted", 51); errcnt--;}
                 */
              q = set_ll_list((yyvsp[(3) - (6)].ll_node),LLNULL,EXPR_LIST);
                 /* r = LLNULL;
                   if($6){
                     r = set_ll_list($6,LLNULL,EXPR_LIST);
                     if($7) r = set_ll_list(r,$7,EXPR_LIST);
                   } else
                     if($7) r = set_ll_list(r,$7,EXPR_LIST);
                 */
	      (yyval.bf_node) = get_bfnd(fi,DVM_REDISTRIBUTE_DIR,SMNULL,q,(yyvsp[(4) - (6)].ll_node),(yyvsp[(6) - (6)].ll_node));;}
    break;

  case 714:
#line 5349 "gram1.y"
    {
                 /* r = LLNULL;
                    if($5){
                      r = set_ll_list($5,LLNULL,EXPR_LIST);
                      if($6) r = set_ll_list(r,$6,EXPR_LIST);
                    } else
                      if($6) r = set_ll_list(r,$6,EXPR_LIST);
                  */
	      (yyval.bf_node) = get_bfnd(fi,DVM_REDISTRIBUTE_DIR,SMNULL,(yyvsp[(8) - (8)].ll_node) ,(yyvsp[(3) - (8)].ll_node),(yyvsp[(5) - (8)].ll_node) );
             ;}
    break;

  case 715:
#line 5377 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 716:
#line 5379 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 717:
#line 5383 "gram1.y"
    {(yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 718:
#line 5385 "gram1.y"
    {(yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 719:
#line 5389 "gram1.y"
    {  PTR_SYMB s;
 
          if(parstate == INEXEC){
            if (!(s = (yyvsp[(1) - (1)].hash_entry)->id_attr))
              {
	         s = make_array((yyvsp[(1) - (1)].hash_entry), TYNULL, LLNULL, 0, LOCAL);
	     	 s->decl = SOFT;
	      } 
            if(s->attr & PROCESSORS_BIT)
              errstr( "Illegal use of PROCESSORS name %s ", s->ident, 53);
            if(s->attr & TASK_BIT)
              errstr( "Illegal use of task array name %s ", s->ident, 71);

          } else {
            s = make_array((yyvsp[(1) - (1)].hash_entry), TYNULL, LLNULL, 0, LOCAL);
            if(s->attr & DISTRIBUTE_BIT)
              errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
            else if( (s->attr & PROCESSORS_BIT) || (s->attr & TASK_BIT) || (s->attr & INHERIT_BIT))
              errstr("Inconsistent declaration of identifier  %s",s->ident, 16);
            else
              s->attr = s->attr | DISTRIBUTE_BIT;
          } 
         if(s->attr & ALIGN_BIT)
               errstr("A distributee may not have the ALIGN attribute:%s",s->ident, 54);
          (yyval.ll_node) = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);               	  
	;}
    break;

  case 720:
#line 5418 "gram1.y"
    {  PTR_SYMB s;
          s = make_array((yyvsp[(1) - (4)].hash_entry), TYNULL, LLNULL, 0, LOCAL);
        
          if(parstate != INEXEC) 
               errstr( "Illegal distributee:%s", s->ident, 312);
          else {
            if(s->attr & PROCESSORS_BIT)
               errstr( "Illegal use of PROCESSORS name %s ", s->ident, 53);  
            if(s->attr & TASK_BIT)
               errstr( "Illegal use of task array name %s ", s->ident, 71);        
            if(s->attr & ALIGN_BIT)
               errstr("A distributee may not have the ALIGN attribute:%s",s->ident, 54);
            if(!(s->attr & DVM_POINTER_BIT))
               errstr("Illegal distributee:%s", s->ident, 312);
          /*s->attr = s->attr | DISTRIBUTE_BIT;*/
	  (yyval.ll_node) = make_llnd(fi,ARRAY_REF, (yyvsp[(3) - (4)].ll_node), LLNULL, s); 
          }
        
	;}
    break;

  case 721:
#line 5441 "gram1.y"
    {  PTR_SYMB s;
          if((s=(yyvsp[(1) - (1)].hash_entry)->id_attr) == SMNULL)
            s = make_array((yyvsp[(1) - (1)].hash_entry), TYNULL, LLNULL, 0, LOCAL);
          if((parstate == INEXEC) && !(s->attr & PROCESSORS_BIT))
               errstr( "'%s' is not processor array ", s->ident, 67);
	  (yyval.symbol) = s;
	;}
    break;

  case 722:
#line 5461 "gram1.y"
    { (yyval.ll_node) = LLNULL;  ;}
    break;

  case 723:
#line 5463 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (2)].ll_node);;}
    break;

  case 724:
#line 5467 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(2) - (3)].ll_node);;}
    break;

  case 725:
#line 5488 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(2) - (2)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 726:
#line 5490 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (4)].ll_node),(yyvsp[(4) - (4)].ll_node),EXPR_LIST); ;}
    break;

  case 727:
#line 5493 "gram1.y"
    { opt_kwd_ = YES; ;}
    break;

  case 728:
#line 5502 "gram1.y"
    {  
               (yyval.ll_node) = make_llnd(fi,BLOCK_OP, LLNULL, LLNULL, SMNULL);
        ;}
    break;

  case 729:
#line 5506 "gram1.y"
    {  err("Distribution format BLOCK(n) is not permitted in FDVM", 55);
          (yyval.ll_node) = make_llnd(fi,BLOCK_OP, (yyvsp[(4) - (5)].ll_node), LLNULL, SMNULL);
          endioctl();
        ;}
    break;

  case 730:
#line 5511 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,BLOCK_OP, LLNULL, LLNULL, (yyvsp[(3) - (4)].symbol)); ;}
    break;

  case 731:
#line 5513 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,BLOCK_OP,  (yyvsp[(5) - (6)].ll_node),  LLNULL,  (yyvsp[(3) - (6)].symbol)); ;}
    break;

  case 732:
#line 5515 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,BLOCK_OP,  LLNULL, (yyvsp[(3) - (4)].ll_node),  SMNULL); ;}
    break;

  case 733:
#line 5517 "gram1.y"
    { 
          (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
          (yyval.ll_node)->entry.string_val = (char *) "*";
          (yyval.ll_node)->type = global_string;
        ;}
    break;

  case 734:
#line 5523 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,INDIRECT_OP, LLNULL, LLNULL, (yyvsp[(3) - (4)].symbol)); ;}
    break;

  case 735:
#line 5525 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,INDIRECT_OP, (yyvsp[(3) - (4)].ll_node), LLNULL, SMNULL); ;}
    break;

  case 736:
#line 5529 "gram1.y"
    {  PTR_SYMB s;
	      s = make_array((yyvsp[(1) - (1)].hash_entry), TYNULL, LLNULL, 0, LOCAL);
              if((s->attr & PROCESSORS_BIT) ||(s->attr & TASK_BIT)  || (s->attr & TEMPLATE_BIT)) 
                errstr("Inconsistent declaration of identifier  %s", s->ident, 16); 
       
	      (yyval.symbol) = s;
	   ;}
    break;

  case 737:
#line 5539 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DERIVED_OP, (yyvsp[(2) - (6)].ll_node), (yyvsp[(6) - (6)].ll_node), SMNULL); ;}
    break;

  case 738:
#line 5543 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 739:
#line 5545 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 740:
#line 5550 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 741:
#line 5552 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), SMNULL);;}
    break;

  case 742:
#line 5556 "gram1.y"
    { 
              (yyval.ll_node) = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, (yyvsp[(1) - (1)].symbol));
	    ;}
    break;

  case 743:
#line 5560 "gram1.y"
    { 
              (yyval.ll_node) = make_llnd(fi,ARRAY_REF, (yyvsp[(3) - (4)].ll_node), LLNULL, (yyvsp[(1) - (4)].symbol));
	    ;}
    break;

  case 744:
#line 5566 "gram1.y"
    { 
              if (!((yyval.symbol) = (yyvsp[(1) - (1)].hash_entry)->id_attr))
              {
	         (yyval.symbol) = make_array((yyvsp[(1) - (1)].hash_entry), TYNULL, LLNULL,0,LOCAL);
	         (yyval.symbol)->decl = SOFT;
	      } 
            ;}
    break;

  case 745:
#line 5576 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 746:
#line 5578 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 747:
#line 5582 "gram1.y"
    {  (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 748:
#line 5584 "gram1.y"
    {  (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 749:
#line 5586 "gram1.y"
    {
                      (yyvsp[(2) - (3)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(3) - (3)].ll_node); 
                      (yyval.ll_node) = (yyvsp[(2) - (3)].ll_node);   
                   ;}
    break;

  case 750:
#line 5593 "gram1.y"
    { PTR_SYMB s;
            s = make_scalar((yyvsp[(1) - (1)].hash_entry),TYNULL,LOCAL);
	    (yyval.ll_node) = make_llnd(fi,DUMMY_REF, LLNULL, LLNULL, s);
            /*$$->type = global_int;*/
          ;}
    break;

  case 751:
#line 5610 "gram1.y"
    {  (yyval.ll_node) = LLNULL; ;}
    break;

  case 752:
#line 5612 "gram1.y"
    {  (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 753:
#line 5616 "gram1.y"
    {  (yyval.ll_node) = set_ll_list((yyvsp[(2) - (2)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 754:
#line 5618 "gram1.y"
    {  (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 755:
#line 5622 "gram1.y"
    {  if((yyvsp[(1) - (1)].ll_node)->type->variant != T_STRING)
                 errstr( "Illegal type of shadow_name", 627);
               (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); 
            ;}
    break;

  case 756:
#line 5629 "gram1.y"
    { char *q;
          nioctl = 1;
          q = (yyvsp[(1) - (2)].ll_node)->entry.string_val;
          if((!strcmp(q,"shadow")) && ((yyvsp[(2) - (2)].ll_node)->variant == INT_VAL))                          (yyval.ll_node) = make_llnd(fi,SPEC_PAIR, (yyvsp[(1) - (2)].ll_node), (yyvsp[(2) - (2)].ll_node), SMNULL);
          else
          {  err("Illegal shadow width specification", 56);
             (yyval.ll_node) = LLNULL;
          }
        ;}
    break;

  case 757:
#line 5639 "gram1.y"
    { char *ql, *qh;
          PTR_LLND p1, p2;
          nioctl = 2;
          ql = (yyvsp[(1) - (5)].ll_node)->entry.string_val;
          qh = (yyvsp[(4) - (5)].ll_node)->entry.string_val;
          if((!strcmp(ql,"low_shadow")) && ((yyvsp[(2) - (5)].ll_node)->variant == INT_VAL) && (!strcmp(qh,"high_shadow")) && ((yyvsp[(5) - (5)].ll_node)->variant == INT_VAL)) 
              {
                 p1 = make_llnd(fi,SPEC_PAIR, (yyvsp[(1) - (5)].ll_node), (yyvsp[(2) - (5)].ll_node), SMNULL);
                 p2 = make_llnd(fi,SPEC_PAIR, (yyvsp[(4) - (5)].ll_node), (yyvsp[(5) - (5)].ll_node), SMNULL);
                 (yyval.ll_node) = make_llnd(fi,CONS, p1, p2, SMNULL);
              } 
          else
          {  err("Illegal shadow width specification", 56);
             (yyval.ll_node) = LLNULL;
          }
        ;}
    break;

  case 758:
#line 5668 "gram1.y"
    { PTR_LLND q;
              q = set_ll_list((yyvsp[(3) - (4)].ll_node),LLNULL,EXPR_LIST);
              (yyval.bf_node) = (yyvsp[(4) - (4)].bf_node);
              (yyval.bf_node)->entry.Template.ll_ptr1 = q;
            ;}
    break;

  case 759:
#line 5683 "gram1.y"
    { PTR_LLND q;
              q = set_ll_list((yyvsp[(3) - (4)].ll_node),LLNULL,EXPR_LIST);
              (yyval.bf_node) = (yyvsp[(4) - (4)].bf_node);
              (yyval.bf_node)->variant = DVM_REALIGN_DIR; 
              (yyval.bf_node)->entry.Template.ll_ptr1 = q;
            ;}
    break;

  case 760:
#line 5690 "gram1.y"
    {
              (yyval.bf_node) = (yyvsp[(3) - (6)].bf_node);
              (yyval.bf_node)->variant = DVM_REALIGN_DIR; 
              (yyval.bf_node)->entry.Template.ll_ptr1 = (yyvsp[(6) - (6)].ll_node);
            ;}
    break;

  case 761:
#line 5708 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 762:
#line 5710 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 763:
#line 5714 "gram1.y"
    {  PTR_SYMB s;
          s = make_array((yyvsp[(1) - (1)].hash_entry), TYNULL, LLNULL, 0, LOCAL);
          if((s->attr & ALIGN_BIT)) 
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
          if((s->attr & PROCESSORS_BIT) || (s->attr & TASK_BIT) || (s->attr & INHERIT_BIT)) 
                errstr( "Inconsistent declaration of identifier  %s", s->ident, 16); 
          else  if(s->attr & DISTRIBUTE_BIT)
               errstr( "An alignee may not have the DISTRIBUTE attribute:'%s'", s->ident,57);             else
                s->attr = s->attr | ALIGN_BIT;     
	  (yyval.ll_node) = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);
	;}
    break;

  case 764:
#line 5728 "gram1.y"
    {PTR_SYMB s;
        s = (yyvsp[(1) - (1)].ll_node)->entry.Template.symbol;
        if(s->attr & PROCESSORS_BIT)
               errstr( "Illegal use of PROCESSORS name %s ", s->ident, 53);
        else  if(s->attr & TASK_BIT)
              errstr( "Illegal use of task array name %s ", s->ident, 71);
        else if( !(s->attr & DIMENSION_BIT) && !(s->attr & DVM_POINTER_BIT))
            errstr("The alignee %s isn't an array", s->ident, 58);
        else {
            /*  if(!(s->attr & DYNAMIC_BIT))
                 errstr("'%s' hasn't the DYNAMIC attribute", s->ident, 59);
             */
              if(!(s->attr & ALIGN_BIT) && !(s->attr & INHERIT_BIT))
                 errstr("'%s' hasn't the ALIGN attribute", s->ident, 60);
              if(s->attr & DISTRIBUTE_BIT)
                 errstr("An alignee may not have the DISTRIBUTE attribute: %s", s->ident, 57);

/*               if(s->entry.var_decl.local == IO)
 *                 errstr("An alignee may not be the dummy argument");
*/
          }
	  (yyval.ll_node) = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);
	;}
    break;

  case 765:
#line 5754 "gram1.y"
    { /* PTR_LLND r;
              if($7) {
                r = set_ll_list($6,LLNULL,EXPR_LIST);
                r = set_ll_list(r,$7,EXPR_LIST);
              }
              else
                r = $6;
              */
            (yyval.bf_node) = get_bfnd(fi,DVM_ALIGN_DIR,SMNULL,LLNULL,(yyvsp[(2) - (6)].ll_node),(yyvsp[(6) - (6)].ll_node));
           ;}
    break;

  case 766:
#line 5767 "gram1.y"
    {
           (yyval.ll_node) = make_llnd(fi,ARRAY_REF, (yyvsp[(3) - (4)].ll_node), LLNULL, (yyvsp[(1) - (4)].symbol));        
          ;}
    break;

  case 767:
#line 5783 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 768:
#line 5785 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 769:
#line 5788 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 770:
#line 5790 "gram1.y"
    {
                  (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
                  (yyval.ll_node)->entry.string_val = (char *) "*";
                  (yyval.ll_node)->type = global_string;
                 ;}
    break;

  case 771:
#line 5796 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 772:
#line 5800 "gram1.y"
    { 
         /* if(parstate == INEXEC){ *for REALIGN directive*
              if (!($$ = $1->id_attr))
              {
	         $$ = make_array($1, TYNULL, LLNULL,0,LOCAL);
	     	 $$->decl = SOFT;
	      } 
          } else
             $$ = make_array($1, TYNULL, LLNULL, 0, LOCAL);
          */
          if (!((yyval.symbol) = (yyvsp[(1) - (1)].hash_entry)->id_attr))
          {
	       (yyval.symbol) = make_array((yyvsp[(1) - (1)].hash_entry), TYNULL, LLNULL,0,LOCAL);
	       (yyval.symbol)->decl = SOFT;
	  } 
          (yyval.symbol)->attr = (yyval.symbol)->attr | ALIGN_BASE_BIT;
          if((yyval.symbol)->attr & PROCESSORS_BIT)
               errstr( "Illegal use of PROCESSORS name %s ", (yyval.symbol)->ident, 53);
          else  if((yyval.symbol)->attr & TASK_BIT)
               errstr( "Illegal use of task array name %s ", (yyval.symbol)->ident, 71);
          else
          if((parstate == INEXEC) /* for  REALIGN directive */
             &&   !((yyval.symbol)->attr & DIMENSION_BIT) && !((yyval.symbol)->attr & DVM_POINTER_BIT))
            errstr("The align-target %s isn't declared as array", (yyval.symbol)->ident, 61); 
         ;}
    break;

  case 773:
#line 5828 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 774:
#line 5830 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 775:
#line 5834 "gram1.y"
    { PTR_SYMB s;
            s = make_scalar((yyvsp[(1) - (1)].hash_entry),TYNULL,LOCAL);
            if(s->type->variant != T_INT || s->attr & PARAMETER_BIT)             
              errstr("The align-dummy %s isn't a scalar integer variable", s->ident, 62); 
	   (yyval.ll_node) = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
           (yyval.ll_node)->type = global_int;
         ;}
    break;

  case 776:
#line 5842 "gram1.y"
    {  
          (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
          (yyval.ll_node)->entry.string_val = (char *) "*";
          (yyval.ll_node)->type = global_string;
        ;}
    break;

  case 777:
#line 5848 "gram1.y"
    {   (yyval.ll_node) = make_llnd(fi,DDOT, LLNULL, LLNULL, SMNULL); ;}
    break;

  case 778:
#line 5851 "gram1.y"
    { PTR_SYMB s;
	             PTR_LLND q, r, p;
                     int numdim;
                     if(type_options & PROCESSORS_BIT) {    /* 27.06.18 || (type_options & TEMPLATE_BIT)){ */
                       if(! explicit_shape) {
                         err("Explicit shape specification is required", 50);
		         /*$$ = BFNULL;*/
	               }
                     } 

                    /*  else {
                       if($6)
                         err("Shape specification is not permitted", 263);
                     } */

                     if(type_options & DIMENSION_BIT)
                       { p = attr_dims; numdim = attr_ndim;}
                     else
                       { p = LLNULL; numdim = 0; }
                     if((yyvsp[(6) - (6)].ll_node))          /*dimension information after the object name*/
                     { p = (yyvsp[(6) - (6)].ll_node); numdim = ndim;} /*overrides the DIMENSION attribute */
	             s = make_array((yyvsp[(5) - (6)].hash_entry), TYNULL, p, numdim, LOCAL);

                     if((type_options & COMMON_BIT) && !(type_options & TEMPLATE_BIT))
                     {
                        err("Illegal combination of attributes", 63);
                        type_options = type_options & (~COMMON_BIT);
                     }
                     if((type_options & PROCESSORS_BIT) &&((type_options & ALIGN_BIT) ||(type_options & DISTRIBUTE_BIT) ||(type_options & TEMPLATE_BIT) || (type_options & DYNAMIC_BIT) ||(type_options & SHADOW_BIT) ))
                        err("Illegal combination of attributes", 63);
                     else  if((type_options & PROCESSORS_BIT) && ((s->attr & ALIGN_BIT) ||(s->attr & DISTRIBUTE_BIT) ||(s->attr & TEMPLATE_BIT) || (s->attr & DYNAMIC_BIT) ||(s->attr & SHADOW_BIT)) )
                     {  errstr("Inconsistent declaration of  %s", s->ident, 16);
                        type_options = type_options & (~PROCESSORS_BIT);
                     }
                     else if ((s->attr & PROCESSORS_BIT) && ((type_options & ALIGN_BIT) ||(type_options & DISTRIBUTE_BIT) ||(type_options & TEMPLATE_BIT) || (type_options & DYNAMIC_BIT) ||(type_options & SHADOW_BIT))) 
                        errstr("Inconsistent declaration of  %s", s->ident, 16);
                     else if ((s->attr & INHERIT_BIT) && ((type_options & ALIGN_BIT) ||(type_options & DISTRIBUTE_BIT)))
                        errstr("Inconsistent declaration of  %s", s->ident, 16);
                     if(( s->attr & DISTRIBUTE_BIT) &&  (type_options & DISTRIBUTE_BIT))
                           errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
                     if(( s->attr & ALIGN_BIT) &&  (type_options & ALIGN_BIT))
                           errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
                     if(( s->attr & SHADOW_BIT) &&  (type_options & SHADOW_BIT))
                           errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
                     if(( s->attr & TEMPLATE_BIT) &&  (type_options & TEMPLATE_BIT))
                           errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
                     if(( s->attr & PROCESSORS_BIT) &&  (type_options & PROCESSORS_BIT))
                           errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
	             s->attr = s->attr | type_options;
                     if((yyvsp[(6) - (6)].ll_node)) s->attr = s->attr | DIMENSION_BIT;
                     if((s->attr & DISTRIBUTE_BIT) && (s->attr & ALIGN_BIT))
                       errstr("%s has the DISTRIBUTE and ALIGN attribute",s->ident, 64);
	             q = make_llnd(fi,ARRAY_REF, (yyvsp[(6) - (6)].ll_node), LLNULL, s);
	             if(p) s->type->entry.ar_decl.ranges = p;
	             r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	             (yyval.bf_node) = get_bfnd(fi,DVM_VAR_DECL, SMNULL, r, LLNULL,(yyvsp[(1) - (6)].ll_node));
	            ;}
    break;

  case 779:
#line 5909 "gram1.y"
    { PTR_SYMB s;
	             PTR_LLND q, r, p;
                     int numdim;
                    if(type_options & PROCESSORS_BIT) { /*23.10.18  || (type_options & TEMPLATE_BIT)){ */
                       if(! explicit_shape) {
                         err("Explicit shape specification is required", 50);
		         /*$$ = BFNULL;*/
	               }
                     } 
                    /* else {
                       if($4)
                         err("Shape specification is not permitted", 263);
                     } */
                     if(type_options & DIMENSION_BIT)
                       { p = attr_dims; numdim = attr_ndim;}
                     else
                       { p = LLNULL; numdim = 0; }
                     if((yyvsp[(4) - (4)].ll_node))                   /*dimension information after the object name*/
                     { p = (yyvsp[(4) - (4)].ll_node); numdim = ndim;}/*overrides the DIMENSION attribute */
	             s = make_array((yyvsp[(3) - (4)].hash_entry), TYNULL, p, numdim, LOCAL);

                     if((type_options & COMMON_BIT) && !(type_options & TEMPLATE_BIT))
                     {
                        err("Illegal combination of attributes", 63);
                        type_options = type_options & (~COMMON_BIT);
                     }
                     if((type_options & PROCESSORS_BIT) &&((type_options & ALIGN_BIT) ||(type_options & DISTRIBUTE_BIT) ||(type_options & TEMPLATE_BIT) || (type_options & DYNAMIC_BIT) ||(type_options & SHADOW_BIT) ))
                       err("Illegal combination of attributes", 63);
                     else  if((type_options & PROCESSORS_BIT) && ((s->attr & ALIGN_BIT) ||(s->attr & DISTRIBUTE_BIT) ||(s->attr & TEMPLATE_BIT) || (s->attr & DYNAMIC_BIT) ||(s->attr & SHADOW_BIT)) )
                     {  errstr("Inconsistent declaration of identifier %s", s->ident, 16);
                        type_options = type_options & (~PROCESSORS_BIT);
                     }
                     else if ((s->attr & PROCESSORS_BIT) && ((type_options & ALIGN_BIT) ||(type_options & DISTRIBUTE_BIT) ||(type_options & TEMPLATE_BIT) || (type_options & DYNAMIC_BIT) ||(type_options & SHADOW_BIT))) 
                          errstr("Inconsistent declaration of identifier  %s", s->ident,16);
                     else if ((s->attr & INHERIT_BIT) && ((type_options & ALIGN_BIT) ||(type_options & DISTRIBUTE_BIT)))
                          errstr("Inconsistent declaration of identifier %s", s->ident, 16);
                     if(( s->attr & DISTRIBUTE_BIT) &&  (type_options & DISTRIBUTE_BIT))
                          errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
                     if(( s->attr & ALIGN_BIT) &&  (type_options & ALIGN_BIT))
                          errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
                     if(( s->attr & SHADOW_BIT) &&  (type_options & SHADOW_BIT))
                          errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
                     if(( s->attr & TEMPLATE_BIT) &&  (type_options & TEMPLATE_BIT))
                          errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
                     if(( s->attr & PROCESSORS_BIT) &&  (type_options & PROCESSORS_BIT))
                          errstr( "Multiple declaration of identifier  %s ", s->ident, 73);   
	             s->attr = s->attr | type_options;
                     if((yyvsp[(4) - (4)].ll_node)) s->attr = s->attr | DIMENSION_BIT;
                     if((s->attr & DISTRIBUTE_BIT) && (s->attr & ALIGN_BIT))
                           errstr("%s has the DISTRIBUTE and ALIGN attribute",s->ident, 64);
	             q = make_llnd(fi,ARRAY_REF, (yyvsp[(4) - (4)].ll_node), LLNULL, s);
	             if(p) s->type->entry.ar_decl.ranges = p;
	             r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	             add_to_lowLevelList(r, (yyvsp[(1) - (4)].bf_node)->entry.Template.ll_ptr1);
	            ;}
    break;

  case 780:
#line 5973 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); type_options = type_opt; ;}
    break;

  case 781:
#line 5975 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (4)].ll_node),(yyvsp[(4) - (4)].ll_node),EXPR_LIST); type_options = type_options | type_opt;;}
    break;

  case 782:
#line 5978 "gram1.y"
    { type_opt = TEMPLATE_BIT;
               (yyval.ll_node) = make_llnd(fi,TEMPLATE_OP,LLNULL,LLNULL,SMNULL);
               ;}
    break;

  case 783:
#line 5982 "gram1.y"
    { type_opt = PROCESSORS_BIT;
                (yyval.ll_node) = make_llnd(fi,PROCESSORS_OP,LLNULL,LLNULL,SMNULL);
               ;}
    break;

  case 784:
#line 5986 "gram1.y"
    { type_opt = PROCESSORS_BIT;
                (yyval.ll_node) = make_llnd(fi,PROCESSORS_OP,LLNULL,LLNULL,SMNULL);
               ;}
    break;

  case 785:
#line 5990 "gram1.y"
    { type_opt = DYNAMIC_BIT;
                (yyval.ll_node) = make_llnd(fi,DYNAMIC_OP,LLNULL,LLNULL,SMNULL);
               ;}
    break;

  case 786:
#line 6007 "gram1.y"
    {
                if(! explicit_shape) {
                  err("Explicit shape specification is required", 50);
                }
                if(! (yyvsp[(3) - (4)].ll_node)) {
                  err("No shape specification", 65);
	        }
                type_opt = DIMENSION_BIT;
                attr_ndim = ndim; attr_dims = (yyvsp[(3) - (4)].ll_node);
                (yyval.ll_node) = make_llnd(fi,DIMENSION_OP,(yyvsp[(3) - (4)].ll_node),LLNULL,SMNULL);
	       ;}
    break;

  case 787:
#line 6019 "gram1.y"
    { type_opt = SHADOW_BIT;
                  (yyval.ll_node) = make_llnd(fi,SHADOW_OP,(yyvsp[(2) - (2)].ll_node),LLNULL,SMNULL);
                 ;}
    break;

  case 788:
#line 6023 "gram1.y"
    { type_opt = ALIGN_BIT;
                  (yyval.ll_node) = make_llnd(fi,ALIGN_OP,(yyvsp[(3) - (7)].ll_node),(yyvsp[(7) - (7)].ll_node),SMNULL);
                 ;}
    break;

  case 789:
#line 6027 "gram1.y"
    { type_opt = ALIGN_BIT;
                  (yyval.ll_node) = make_llnd(fi,ALIGN_OP,LLNULL,SMNULL,SMNULL);
                ;}
    break;

  case 790:
#line 6037 "gram1.y"
    { 
                 type_opt = DISTRIBUTE_BIT;
                 (yyval.ll_node) = make_llnd(fi,DISTRIBUTE_OP,(yyvsp[(2) - (4)].ll_node),(yyvsp[(4) - (4)].ll_node),SMNULL);
                ;}
    break;

  case 791:
#line 6042 "gram1.y"
    { 
                 type_opt = DISTRIBUTE_BIT;
                 (yyval.ll_node) = make_llnd(fi,DISTRIBUTE_OP,LLNULL,LLNULL,SMNULL);
                ;}
    break;

  case 792:
#line 6047 "gram1.y"
    {
                 type_opt = COMMON_BIT;
                 (yyval.ll_node) = make_llnd(fi,COMMON_OP, LLNULL, LLNULL, SMNULL);
                ;}
    break;

  case 793:
#line 6054 "gram1.y"
    { 
	      PTR_LLND  l;
	      l = make_llnd(fi, TYPE_OP, LLNULL, LLNULL, SMNULL);
	      l->type = (yyvsp[(1) - (11)].data_type);
	      (yyval.bf_node) = get_bfnd(fi,DVM_POINTER_DIR, SMNULL, (yyvsp[(11) - (11)].ll_node),(yyvsp[(7) - (11)].ll_node), l);
	    ;}
    break;

  case 794:
#line 6062 "gram1.y"
    {ndim = 0;;}
    break;

  case 795:
#line 6063 "gram1.y"
    { PTR_LLND  q;
             if(ndim == maxdim)
		err("Too many dimensions", 43);
	      else if(ndim < maxdim)
		q = make_llnd(fi,DDOT,LLNULL,LLNULL,SMNULL);
	      ++ndim;
              (yyval.ll_node) = set_ll_list(q, LLNULL, EXPR_LIST);
	       /*$$ = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);*/
	       /*$$->type = global_default;*/
	    ;}
    break;

  case 796:
#line 6074 "gram1.y"
    { PTR_LLND  q;
             if(ndim == maxdim)
		err("Too many dimensions", 43);
	      else if(ndim < maxdim)
		q = make_llnd(fi,DDOT,LLNULL,LLNULL,SMNULL);
	      ++ndim;
              (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), q, EXPR_LIST);
            ;}
    break;

  case 797:
#line 6085 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 798:
#line 6087 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 799:
#line 6091 "gram1.y"
    {PTR_SYMB s;
           /* s = make_scalar($1,TYNULL,LOCAL);*/
            s = make_array((yyvsp[(1) - (1)].hash_entry),TYNULL,LLNULL,0,LOCAL);
            s->attr = s->attr | DVM_POINTER_BIT;
            if((s->attr & PROCESSORS_BIT) || (s->attr & TASK_BIT) || (s->attr & INHERIT_BIT))
               errstr( "Inconsistent declaration of identifier %s", s->ident, 16);     
            (yyval.ll_node) = make_llnd(fi,VAR_REF,LLNULL,LLNULL,s);
            ;}
    break;

  case 800:
#line 6102 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_HEAP_DIR, SMNULL, (yyvsp[(3) - (3)].ll_node), LLNULL, LLNULL);;}
    break;

  case 801:
#line 6106 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 802:
#line 6108 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 803:
#line 6112 "gram1.y"
    {  PTR_SYMB s;
	      s = make_array((yyvsp[(1) - (1)].hash_entry), TYNULL, LLNULL, 0, LOCAL);
              s->attr = s->attr | HEAP_BIT;
              if((s->attr & PROCESSORS_BIT) ||(s->attr & TASK_BIT)  || (s->attr & TEMPLATE_BIT) || (s->attr & ALIGN_BIT) || (s->attr & DISTRIBUTE_BIT) || (s->attr & INHERIT_BIT) || (s->attr & DYNAMIC_BIT) || (s->attr & SHADOW_BIT) || (s->attr & DVM_POINTER_BIT)) 
                errstr("Inconsistent declaration of identifier  %s", s->ident, 16); 
      
	      (yyval.ll_node) = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);
	   ;}
    break;

  case 804:
#line 6123 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_CONSISTENT_DIR, SMNULL, (yyvsp[(3) - (3)].ll_node), LLNULL, LLNULL);;}
    break;

  case 805:
#line 6127 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 806:
#line 6129 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 807:
#line 6133 "gram1.y"
    {  PTR_SYMB s;
	      s = make_array((yyvsp[(1) - (1)].hash_entry), TYNULL, LLNULL, 0, LOCAL);
              s->attr = s->attr | CONSISTENT_BIT;
              if((s->attr & PROCESSORS_BIT) ||(s->attr & TASK_BIT)  || (s->attr & TEMPLATE_BIT) || (s->attr & ALIGN_BIT) || (s->attr & DISTRIBUTE_BIT) || (s->attr & INHERIT_BIT) || (s->attr & DYNAMIC_BIT) || (s->attr & SHADOW_BIT) || (s->attr & DVM_POINTER_BIT)) 
                errstr("Inconsistent declaration of identifier  %s", s->ident, 16); 
      
	      (yyval.ll_node) = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);
	   ;}
    break;

  case 808:
#line 6145 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_ASYNCID_DIR, SMNULL, (yyvsp[(3) - (3)].ll_node), LLNULL, LLNULL);;}
    break;

  case 809:
#line 6147 "gram1.y"
    { PTR_LLND p;
              p = make_llnd(fi,COMM_LIST, LLNULL, LLNULL, SMNULL);              
              (yyval.bf_node) = get_bfnd(fi,DVM_ASYNCID_DIR, SMNULL, (yyvsp[(8) - (8)].ll_node), p, LLNULL);
            ;}
    break;

  case 810:
#line 6154 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 811:
#line 6156 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 812:
#line 6160 "gram1.y"
    {  PTR_SYMB s;
              if((yyvsp[(2) - (2)].ll_node)){
                  s = make_array((yyvsp[(1) - (2)].hash_entry), global_default, (yyvsp[(2) - (2)].ll_node), ndim, LOCAL);
		  s->variant = ASYNC_ID;
                  s->attr = s->attr | DIMENSION_BIT;
                  s->type->entry.ar_decl.ranges = (yyvsp[(2) - (2)].ll_node);
                  (yyval.ll_node) = make_llnd(fi,ARRAY_REF, (yyvsp[(2) - (2)].ll_node), LLNULL, s);
              } else {
              s = make_local_entity((yyvsp[(1) - (2)].hash_entry), ASYNC_ID, global_default, LOCAL);
	      (yyval.ll_node) = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
              }
	   ;}
    break;

  case 813:
#line 6176 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_NEW_VALUE_DIR,SMNULL, LLNULL, LLNULL,LLNULL);;}
    break;

  case 814:
#line 6186 "gram1.y"
    {  if((yyvsp[(6) - (7)].ll_node) &&  (yyvsp[(6) - (7)].ll_node)->entry.Template.symbol->attr & TASK_BIT)
                        (yyval.bf_node) = get_bfnd(fi,DVM_PARALLEL_TASK_DIR,SMNULL,(yyvsp[(6) - (7)].ll_node),(yyvsp[(7) - (7)].ll_node),(yyvsp[(4) - (7)].ll_node));
                    else
                        (yyval.bf_node) = get_bfnd(fi,DVM_PARALLEL_ON_DIR,SMNULL,(yyvsp[(6) - (7)].ll_node),(yyvsp[(7) - (7)].ll_node),(yyvsp[(4) - (7)].ll_node));
                 ;}
    break;

  case 815:
#line 6195 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 816:
#line 6197 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 817:
#line 6201 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(3) - (3)].ll_node);;}
    break;

  case 818:
#line 6204 "gram1.y"
    { (yyval.ll_node) = LLNULL; opt_kwd_ = NO;;}
    break;

  case 819:
#line 6209 "gram1.y"
    {
          if((yyvsp[(1) - (4)].ll_node)->type->variant != T_ARRAY) 
             errstr("'%s' isn't array", (yyvsp[(1) - (4)].ll_node)->entry.Template.symbol->ident, 66);
          (yyvsp[(1) - (4)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(3) - (4)].ll_node);
          (yyval.ll_node) = (yyvsp[(1) - (4)].ll_node);
          (yyval.ll_node)->type = (yyvsp[(1) - (4)].ll_node)->type->entry.ar_decl.base_type;
        ;}
    break;

  case 820:
#line 6219 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 821:
#line 6221 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 822:
#line 6225 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 823:
#line 6227 "gram1.y"
    {
             (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
             (yyval.ll_node)->entry.string_val = (char *) "*";
             (yyval.ll_node)->type = global_string;
            ;}
    break;

  case 824:
#line 6235 "gram1.y"
    {  (yyval.ll_node) = LLNULL;;}
    break;

  case 825:
#line 6237 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 826:
#line 6241 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 827:
#line 6243 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (2)].ll_node),(yyvsp[(2) - (2)].ll_node),EXPR_LIST); ;}
    break;

  case 839:
#line 6261 "gram1.y"
    { if((yyvsp[(5) - (8)].symbol)->attr & INDIRECT_BIT)
                            errstr("'%s' is not remote group name", (yyvsp[(5) - (8)].symbol)->ident, 68);
                          (yyval.ll_node) = make_llnd(fi,REMOTE_ACCESS_OP,(yyvsp[(7) - (8)].ll_node),LLNULL,(yyvsp[(5) - (8)].symbol));
                        ;}
    break;

  case 840:
#line 6266 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,REMOTE_ACCESS_OP,(yyvsp[(5) - (6)].ll_node),LLNULL,SMNULL);;}
    break;

  case 841:
#line 6270 "gram1.y"
    {
                          (yyval.ll_node) = make_llnd(fi,CONSISTENT_OP,(yyvsp[(7) - (8)].ll_node),LLNULL,(yyvsp[(5) - (8)].symbol));
                        ;}
    break;

  case 842:
#line 6274 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,CONSISTENT_OP,(yyvsp[(5) - (6)].ll_node),LLNULL,SMNULL);;}
    break;

  case 843:
#line 6278 "gram1.y"
    {  
            if(((yyval.symbol)=(yyvsp[(1) - (1)].hash_entry)->id_attr) == SMNULL){
                errstr("'%s' is not declared as group", (yyvsp[(1) - (1)].hash_entry)->ident, 74);
                (yyval.symbol) = make_local_entity((yyvsp[(1) - (1)].hash_entry),CONSISTENT_GROUP_NAME,global_default,LOCAL);
            } else {
                if((yyval.symbol)->variant != CONSISTENT_GROUP_NAME)
                   errstr("'%s' is not declared as group", (yyvsp[(1) - (1)].hash_entry)->ident, 74);
            }
          ;}
    break;

  case 844:
#line 6291 "gram1.y"
    {(yyval.ll_node) = make_llnd(fi,NEW_SPEC_OP,(yyvsp[(5) - (6)].ll_node),LLNULL,SMNULL);;}
    break;

  case 845:
#line 6295 "gram1.y"
    {(yyval.ll_node) = make_llnd(fi,NEW_SPEC_OP,(yyvsp[(5) - (6)].ll_node),LLNULL,SMNULL);;}
    break;

  case 846:
#line 6299 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_PRIVATE_OP,(yyvsp[(5) - (6)].ll_node),LLNULL,SMNULL);;}
    break;

  case 847:
#line 6303 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_CUDA_BLOCK_OP,(yyvsp[(5) - (6)].ll_node),LLNULL,SMNULL);;}
    break;

  case 848:
#line 6306 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST);;}
    break;

  case 849:
#line 6308 "gram1.y"
    {(yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST);;}
    break;

  case 850:
#line 6310 "gram1.y"
    {(yyval.ll_node) = set_ll_list((yyvsp[(1) - (5)].ll_node),(yyvsp[(3) - (5)].ll_node),EXPR_LIST); (yyval.ll_node) = set_ll_list((yyval.ll_node),(yyvsp[(5) - (5)].ll_node),EXPR_LIST);;}
    break;

  case 851:
#line 6314 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST);;}
    break;

  case 852:
#line 6316 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST);;}
    break;

  case 853:
#line 6320 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_TIE_OP,(yyvsp[(5) - (6)].ll_node),LLNULL,SMNULL);;}
    break;

  case 854:
#line 6324 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST);;}
    break;

  case 855:
#line 6326 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST);;}
    break;

  case 856:
#line 6330 "gram1.y"
    { if(!((yyvsp[(5) - (8)].symbol)->attr & INDIRECT_BIT))
                         errstr("'%s' is not indirect group name", (yyvsp[(5) - (8)].symbol)->ident, 313);
                      (yyval.ll_node) = make_llnd(fi,INDIRECT_ACCESS_OP,(yyvsp[(7) - (8)].ll_node),LLNULL,(yyvsp[(5) - (8)].symbol));
                    ;}
    break;

  case 857:
#line 6335 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,INDIRECT_ACCESS_OP,(yyvsp[(5) - (6)].ll_node),LLNULL,SMNULL);;}
    break;

  case 858:
#line 6339 "gram1.y"
    {(yyval.ll_node) = make_llnd(fi,STAGE_OP,(yyvsp[(5) - (6)].ll_node),LLNULL,SMNULL);;}
    break;

  case 859:
#line 6343 "gram1.y"
    {(yyval.ll_node) = make_llnd(fi,ACROSS_OP,(yyvsp[(4) - (4)].ll_node),LLNULL,SMNULL);;}
    break;

  case 860:
#line 6345 "gram1.y"
    {(yyval.ll_node) = make_llnd(fi,ACROSS_OP,(yyvsp[(4) - (5)].ll_node),(yyvsp[(5) - (5)].ll_node),SMNULL);;}
    break;

  case 861:
#line 6349 "gram1.y"
    {  if((yyvsp[(3) - (5)].ll_node))
                     (yyval.ll_node) = make_llnd(fi,DDOT,(yyvsp[(3) - (5)].ll_node),(yyvsp[(4) - (5)].ll_node),SMNULL);
                   else
                     (yyval.ll_node) = (yyvsp[(4) - (5)].ll_node);
                ;}
    break;

  case 862:
#line 6357 "gram1.y"
    { opt_in_out = YES; ;}
    break;

  case 863:
#line 6361 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "in";
              (yyval.ll_node)->type = global_string;
            ;}
    break;

  case 864:
#line 6367 "gram1.y"
    {
	      (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "out";
              (yyval.ll_node)->type = global_string;
            ;}
    break;

  case 865:
#line 6373 "gram1.y"
    {  (yyval.ll_node) = LLNULL; opt_in_out = NO;;}
    break;

  case 866:
#line 6377 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST);;}
    break;

  case 867:
#line 6379 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST);;}
    break;

  case 868:
#line 6383 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 869:
#line 6385 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (4)].ll_node);
                    (yyval.ll_node)-> entry.Template.ll_ptr1 = (yyvsp[(3) - (4)].ll_node);  
                  ;}
    break;

  case 870:
#line 6389 "gram1.y"
    { /*  PTR_LLND p;
                       p = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
                       p->entry.string_val = (char *) "corner";
                       p->type = global_string;
                   */
                   (yyvsp[(1) - (7)].ll_node)-> entry.Template.ll_ptr1 = (yyvsp[(3) - (7)].ll_node);  
                   (yyval.ll_node) = make_llnd(fi,ARRAY_OP,(yyvsp[(1) - (7)].ll_node),(yyvsp[(6) - (7)].ll_node),SMNULL);
                 ;}
    break;

  case 871:
#line 6401 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST);;}
    break;

  case 872:
#line 6403 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST);;}
    break;

  case 873:
#line 6407 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), SMNULL);;}
    break;

  case 874:
#line 6411 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST);;}
    break;

  case 875:
#line 6413 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST);;}
    break;

  case 876:
#line 6417 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT,(yyvsp[(1) - (5)].ll_node),make_llnd(fi,DDOT,(yyvsp[(3) - (5)].ll_node),(yyvsp[(5) - (5)].ll_node),SMNULL),SMNULL); ;}
    break;

  case 877:
#line 6419 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT,(yyvsp[(1) - (3)].ll_node),make_llnd(fi,DDOT,(yyvsp[(3) - (3)].ll_node),LLNULL,SMNULL),SMNULL); ;}
    break;

  case 878:
#line 6421 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT,(yyvsp[(1) - (3)].ll_node),make_llnd(fi,DDOT,LLNULL,(yyvsp[(3) - (3)].ll_node),SMNULL),SMNULL); ;}
    break;

  case 879:
#line 6423 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT,(yyvsp[(1) - (1)].ll_node),LLNULL,SMNULL); ;}
    break;

  case 880:
#line 6425 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT,LLNULL,make_llnd(fi,DDOT,(yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),SMNULL),SMNULL); ;}
    break;

  case 881:
#line 6427 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT,LLNULL,make_llnd(fi,DDOT,(yyvsp[(1) - (1)].ll_node),LLNULL,SMNULL),SMNULL); ;}
    break;

  case 882:
#line 6429 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT,LLNULL,make_llnd(fi,DDOT,LLNULL,(yyvsp[(1) - (1)].ll_node),SMNULL),SMNULL); ;}
    break;

  case 883:
#line 6433 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(3) - (3)].ll_node);;}
    break;

  case 884:
#line 6437 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(3) - (3)].ll_node);;}
    break;

  case 885:
#line 6441 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(3) - (3)].ll_node);;}
    break;

  case 886:
#line 6445 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(2) - (3)].ll_node);;}
    break;

  case 887:
#line 6449 "gram1.y"
    {PTR_LLND q;
                /* q = set_ll_list($9,$6,EXPR_LIST); */
                 q = set_ll_list((yyvsp[(6) - (10)].ll_node),LLNULL,EXPR_LIST); /*podd 11.10.01*/
                 q = add_to_lowLevelList((yyvsp[(9) - (10)].ll_node),q);        /*podd 11.10.01*/
                 (yyval.ll_node) = make_llnd(fi,REDUCTION_OP,q,LLNULL,SMNULL);
                ;}
    break;

  case 888:
#line 6456 "gram1.y"
    {PTR_LLND q;
                 q = set_ll_list((yyvsp[(6) - (8)].ll_node),LLNULL,EXPR_LIST);
                 (yyval.ll_node) = make_llnd(fi,REDUCTION_OP,q,LLNULL,SMNULL);
                ;}
    break;

  case 889:
#line 6462 "gram1.y"
    {  (yyval.ll_node) = make_llnd(fi,REDUCTION_OP,(yyvsp[(9) - (10)].ll_node),LLNULL,(yyvsp[(6) - (10)].symbol)); ;}
    break;

  case 890:
#line 6466 "gram1.y"
    { opt_kwd_r = YES; ;}
    break;

  case 891:
#line 6469 "gram1.y"
    { opt_kwd_r = NO; ;}
    break;

  case 892:
#line 6473 "gram1.y"
    { 
                  if(((yyval.symbol)=(yyvsp[(1) - (1)].hash_entry)->id_attr) == SMNULL) {
                      errstr("'%s' is not declared as reduction group", (yyvsp[(1) - (1)].hash_entry)->ident, 69);
                      (yyval.symbol) = make_local_entity((yyvsp[(1) - (1)].hash_entry),REDUCTION_GROUP_NAME,global_default,LOCAL);
                  } else {
                    if((yyval.symbol)->variant != REDUCTION_GROUP_NAME)
                      errstr("'%s' is not declared as reduction group", (yyvsp[(1) - (1)].hash_entry)->ident, 69);
                  }
                ;}
    break;

  case 893:
#line 6486 "gram1.y"
    {(yyval.ll_node) = set_ll_list((yyvsp[(2) - (2)].ll_node),LLNULL,EXPR_LIST);;}
    break;

  case 894:
#line 6488 "gram1.y"
    {(yyval.ll_node) = set_ll_list((yyvsp[(1) - (4)].ll_node),(yyvsp[(4) - (4)].ll_node),EXPR_LIST);;}
    break;

  case 895:
#line 6492 "gram1.y"
    {(yyval.ll_node) = make_llnd(fi,ARRAY_OP,(yyvsp[(1) - (4)].ll_node),(yyvsp[(3) - (4)].ll_node),SMNULL);;}
    break;

  case 896:
#line 6494 "gram1.y"
    {(yyvsp[(3) - (6)].ll_node) = set_ll_list((yyvsp[(3) - (6)].ll_node),(yyvsp[(5) - (6)].ll_node),EXPR_LIST);
            (yyval.ll_node) = make_llnd(fi,ARRAY_OP,(yyvsp[(1) - (6)].ll_node),(yyvsp[(3) - (6)].ll_node),SMNULL);;}
    break;

  case 897:
#line 6499 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "sum";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 898:
#line 6505 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "product";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 899:
#line 6511 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "min";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 900:
#line 6517 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "max";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 901:
#line 6523 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "or";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 902:
#line 6529 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "and";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 903:
#line 6535 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "eqv";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 904:
#line 6541 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "neqv";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 905:
#line 6547 "gram1.y"
    { err("Illegal reduction operation name", 70);
               errcnt--;
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "unknown";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 906:
#line 6556 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "maxloc";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 907:
#line 6562 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "minloc";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 908:
#line 6579 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SHADOW_RENEW_OP,(yyvsp[(5) - (6)].ll_node),LLNULL,SMNULL);;}
    break;

  case 909:
#line 6587 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SHADOW_START_OP,LLNULL,LLNULL,(yyvsp[(4) - (4)].symbol));;}
    break;

  case 910:
#line 6595 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SHADOW_WAIT_OP,LLNULL,LLNULL,(yyvsp[(4) - (4)].symbol));;}
    break;

  case 911:
#line 6597 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SHADOW_COMP_OP,LLNULL,LLNULL,SMNULL);;}
    break;

  case 912:
#line 6599 "gram1.y"
    {  (yyvsp[(5) - (9)].ll_node)-> entry.Template.ll_ptr1 = (yyvsp[(7) - (9)].ll_node); (yyval.ll_node) = make_llnd(fi,SHADOW_COMP_OP,(yyvsp[(5) - (9)].ll_node),LLNULL,SMNULL);;}
    break;

  case 913:
#line 6603 "gram1.y"
    {(yyval.symbol) = make_local_entity((yyvsp[(1) - (1)].hash_entry), SHADOW_GROUP_NAME,global_default,LOCAL);;}
    break;

  case 914:
#line 6607 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST);;}
    break;

  case 915:
#line 6609 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST);;}
    break;

  case 916:
#line 6613 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 917:
#line 6615 "gram1.y"
    { PTR_LLND p;
          p = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
          p->entry.string_val = (char *) "corner";
          p->type = global_string;
          (yyval.ll_node) = make_llnd(fi,ARRAY_OP,(yyvsp[(1) - (5)].ll_node),p,SMNULL);
         ;}
    break;

  case 918:
#line 6623 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (5)].ll_node);
          (yyval.ll_node)-> entry.Template.ll_ptr1 = (yyvsp[(4) - (5)].ll_node);  
        ;}
    break;

  case 919:
#line 6627 "gram1.y"
    { PTR_LLND p;
          p = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
          p->entry.string_val = (char *) "corner";
          p->type = global_string;
          (yyvsp[(1) - (9)].ll_node)-> entry.Template.ll_ptr1 = (yyvsp[(4) - (9)].ll_node);  
          (yyval.ll_node) = make_llnd(fi,ARRAY_OP,(yyvsp[(1) - (9)].ll_node),p,SMNULL);
       ;}
    break;

  case 920:
#line 6638 "gram1.y"
    { optcorner = YES; ;}
    break;

  case 921:
#line 6642 "gram1.y"
    { PTR_SYMB s;
         s = (yyvsp[(1) - (1)].ll_node)->entry.Template.symbol;
         if(s->attr & PROCESSORS_BIT)
             errstr( "Illegal use of PROCESSORS name %s ", s->ident, 53);
         else if(s->attr & TASK_BIT)
             errstr( "Illegal use of task array name %s ", s->ident, 71);
         else
           if(s->type->variant != T_ARRAY) 
             errstr("'%s' isn't array", s->ident, 66);
           else 
              if((!(s->attr & DISTRIBUTE_BIT)) && (!(s->attr & ALIGN_BIT)))
               ; /*errstr("hpf.gram: %s is not distributed array", s->ident);*/
                
         (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);
        ;}
    break;

  case 922:
#line 6660 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 923:
#line 6662 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 924:
#line 6666 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_SHADOW_START_DIR,(yyvsp[(3) - (3)].symbol),LLNULL,LLNULL,LLNULL);;}
    break;

  case 925:
#line 6668 "gram1.y"
    {errstr("Missing DVM directive prefix", 49);;}
    break;

  case 926:
#line 6672 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_SHADOW_WAIT_DIR,(yyvsp[(3) - (3)].symbol),LLNULL,LLNULL,LLNULL);;}
    break;

  case 927:
#line 6674 "gram1.y"
    {errstr("Missing DVM directive prefix", 49);;}
    break;

  case 928:
#line 6678 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_SHADOW_GROUP_DIR,(yyvsp[(3) - (6)].symbol),(yyvsp[(5) - (6)].ll_node),LLNULL,LLNULL);;}
    break;

  case 929:
#line 6682 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_REDUCTION_START_DIR,(yyvsp[(3) - (3)].symbol),LLNULL,LLNULL,LLNULL);;}
    break;

  case 930:
#line 6686 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_REDUCTION_WAIT_DIR,(yyvsp[(3) - (3)].symbol),LLNULL,LLNULL,LLNULL);;}
    break;

  case 931:
#line 6695 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_CONSISTENT_START_DIR,(yyvsp[(3) - (3)].symbol),LLNULL,LLNULL,LLNULL);;}
    break;

  case 932:
#line 6699 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_CONSISTENT_WAIT_DIR,(yyvsp[(3) - (3)].symbol),LLNULL,LLNULL,LLNULL);;}
    break;

  case 933:
#line 6703 "gram1.y"
    { if(((yyvsp[(4) - (7)].symbol)->attr & INDIRECT_BIT))
                errstr("'%s' is not remote group name", (yyvsp[(4) - (7)].symbol)->ident, 68);
           (yyval.bf_node) = get_bfnd(fi,DVM_REMOTE_ACCESS_DIR,(yyvsp[(4) - (7)].symbol),(yyvsp[(6) - (7)].ll_node),LLNULL,LLNULL);
         ;}
    break;

  case 934:
#line 6708 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_REMOTE_ACCESS_DIR,SMNULL,(yyvsp[(4) - (5)].ll_node),LLNULL,LLNULL);;}
    break;

  case 935:
#line 6712 "gram1.y"
    {  
            if(((yyval.symbol)=(yyvsp[(1) - (1)].hash_entry)->id_attr) == SMNULL){
                errstr("'%s' is not declared as group", (yyvsp[(1) - (1)].hash_entry)->ident, 74);
                (yyval.symbol) = make_local_entity((yyvsp[(1) - (1)].hash_entry),REF_GROUP_NAME,global_default,LOCAL);
            } else {
              if((yyval.symbol)->variant != REF_GROUP_NAME)
                errstr("'%s' is not declared as group", (yyvsp[(1) - (1)].hash_entry)->ident, 74);
            }
          ;}
    break;

  case 936:
#line 6724 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 937:
#line 6726 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 938:
#line 6730 "gram1.y"
    {
              (yyval.ll_node) = (yyvsp[(1) - (4)].ll_node);
              (yyval.ll_node)->entry.Template.ll_ptr1 = (yyvsp[(3) - (4)].ll_node);
            ;}
    break;

  case 939:
#line 6735 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 940:
#line 6739 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 941:
#line 6741 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 942:
#line 6745 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 943:
#line 6747 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT, LLNULL, LLNULL, SMNULL);;}
    break;

  case 944:
#line 6751 "gram1.y"
    {  PTR_LLND q;
             q = make_llnd(fi,EXPR_LIST, (yyvsp[(3) - (3)].ll_node), LLNULL, SMNULL);
             (yyval.bf_node) = get_bfnd(fi,DVM_TASK_DIR,SMNULL,q,LLNULL,LLNULL);
          ;}
    break;

  case 945:
#line 6756 "gram1.y"
    {   PTR_LLND q;
              q = make_llnd(fi,EXPR_LIST, (yyvsp[(3) - (3)].ll_node), LLNULL, SMNULL);
	      add_to_lowLevelList(q, (yyvsp[(1) - (3)].bf_node)->entry.Template.ll_ptr1);
          ;}
    break;

  case 946:
#line 6763 "gram1.y"
    { 
             PTR_SYMB s;
	      s = make_array((yyvsp[(1) - (2)].hash_entry), global_int, (yyvsp[(2) - (2)].ll_node), ndim, LOCAL);
              if((yyvsp[(2) - (2)].ll_node)){
                  s->attr = s->attr | DIMENSION_BIT;
                  s->type->entry.ar_decl.ranges = (yyvsp[(2) - (2)].ll_node);
              }
              else
                  err("No dimensions in TASK directive", 75);
              if(ndim > 1)
                  errstr("Illegal rank of '%s'", s->ident, 76);
              if(s->attr & TASK_BIT)
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
              if((s->attr & ALIGN_BIT) ||(s->attr & DISTRIBUTE_BIT) ||(s->attr & TEMPLATE_BIT) || (s->attr & DYNAMIC_BIT) ||(s->attr & SHADOW_BIT) || (s->attr & PROCESSORS_BIT)  || (s->attr & DVM_POINTER_BIT) || (s->attr & INHERIT_BIT))
                errstr("Inconsistent declaration of identifier  %s", s->ident, 16);
              else
	        s->attr = s->attr | TASK_BIT;
    
	      (yyval.ll_node) = make_llnd(fi,ARRAY_REF, (yyvsp[(2) - (2)].ll_node), LLNULL, s);	  
	    ;}
    break;

  case 947:
#line 6786 "gram1.y"
    {(yyval.bf_node) = get_bfnd(fi,DVM_TASK_REGION_DIR,(yyvsp[(3) - (3)].symbol),LLNULL,LLNULL,LLNULL);;}
    break;

  case 948:
#line 6788 "gram1.y"
    {(yyval.bf_node) = get_bfnd(fi,DVM_TASK_REGION_DIR,(yyvsp[(3) - (4)].symbol),(yyvsp[(4) - (4)].ll_node),LLNULL,LLNULL);;}
    break;

  case 949:
#line 6790 "gram1.y"
    {(yyval.bf_node) = get_bfnd(fi,DVM_TASK_REGION_DIR,(yyvsp[(3) - (4)].symbol),LLNULL,(yyvsp[(4) - (4)].ll_node),LLNULL);;}
    break;

  case 950:
#line 6792 "gram1.y"
    {(yyval.bf_node) = get_bfnd(fi,DVM_TASK_REGION_DIR,(yyvsp[(3) - (5)].symbol),(yyvsp[(4) - (5)].ll_node),(yyvsp[(5) - (5)].ll_node),LLNULL);;}
    break;

  case 951:
#line 6794 "gram1.y"
    {(yyval.bf_node) = get_bfnd(fi,DVM_TASK_REGION_DIR,(yyvsp[(3) - (5)].symbol),(yyvsp[(5) - (5)].ll_node),(yyvsp[(4) - (5)].ll_node),LLNULL);;}
    break;

  case 952:
#line 6798 "gram1.y"
    { PTR_SYMB s;
              if((s=(yyvsp[(1) - (1)].hash_entry)->id_attr) == SMNULL)
                s = make_array((yyvsp[(1) - (1)].hash_entry), TYNULL, LLNULL, 0, LOCAL);
              
              if(!(s->attr & TASK_BIT))
                 errstr("'%s' is not task array", s->ident, 77);
              (yyval.symbol) = s;
              ;}
    break;

  case 953:
#line 6809 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_END_TASK_REGION_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 954:
#line 6813 "gram1.y"
    {  PTR_SYMB s;
              PTR_LLND q;
             /*
              s = make_array($1, TYNULL, LLNULL, 0, LOCAL);                           
	      if((parstate == INEXEC) && !(s->attr & TASK_BIT))
                 errstr("'%s' is not task array", s->ident, 77);  
              q =  set_ll_list($3,LLNULL,EXPR_LIST);
	      $$ = make_llnd(fi,ARRAY_REF, q, LLNULL, s);
              */

              s = (yyvsp[(1) - (4)].symbol);
              q =  set_ll_list((yyvsp[(3) - (4)].ll_node),LLNULL,EXPR_LIST);
	      (yyval.ll_node) = make_llnd(fi,ARRAY_REF, q, LLNULL, s);
	   ;}
    break;

  case 955:
#line 6828 "gram1.y"
    {  PTR_LLND q; 
              q =  set_ll_list((yyvsp[(3) - (4)].ll_node),LLNULL,EXPR_LIST);
	      (yyval.ll_node) = make_llnd(fi,ARRAY_REF, q, LLNULL, (yyvsp[(1) - (4)].symbol));
	   ;}
    break;

  case 956:
#line 6835 "gram1.y"
    {              
         (yyval.bf_node) = get_bfnd(fi,DVM_ON_DIR,SMNULL,(yyvsp[(3) - (4)].ll_node),(yyvsp[(4) - (4)].ll_node),LLNULL);
    ;}
    break;

  case 957:
#line 6841 "gram1.y"
    {(yyval.ll_node) = LLNULL;;}
    break;

  case 958:
#line 6843 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 959:
#line 6847 "gram1.y"
    {(yyval.bf_node) = get_bfnd(fi,DVM_END_ON_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 960:
#line 6851 "gram1.y"
    { PTR_LLND q;
        /* if(!($6->attr & PROCESSORS_BIT))
           errstr("'%s' is not processor array", $6->ident, 67);
         */
        q = make_llnd(fi,ARRAY_REF, (yyvsp[(7) - (7)].ll_node), LLNULL, (yyvsp[(6) - (7)].symbol));
        (yyval.bf_node) = get_bfnd(fi,DVM_MAP_DIR,SMNULL,(yyvsp[(3) - (7)].ll_node),q,LLNULL);
      ;}
    break;

  case 961:
#line 6859 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_MAP_DIR,SMNULL,(yyvsp[(3) - (6)].ll_node),LLNULL,(yyvsp[(6) - (6)].ll_node)); ;}
    break;

  case 962:
#line 6863 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_PREFETCH_DIR,(yyvsp[(3) - (3)].symbol),LLNULL,LLNULL,LLNULL);;}
    break;

  case 963:
#line 6867 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_RESET_DIR,(yyvsp[(3) - (3)].symbol),LLNULL,LLNULL,LLNULL);;}
    break;

  case 964:
#line 6875 "gram1.y"
    { if(!((yyvsp[(4) - (7)].symbol)->attr & INDIRECT_BIT))
                         errstr("'%s' is not indirect group name", (yyvsp[(4) - (7)].symbol)->ident, 313);
                      (yyval.bf_node) = get_bfnd(fi,DVM_INDIRECT_ACCESS_DIR,(yyvsp[(4) - (7)].symbol),(yyvsp[(6) - (7)].ll_node),LLNULL,LLNULL);
                    ;}
    break;

  case 965:
#line 6880 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_INDIRECT_ACCESS_DIR,SMNULL,(yyvsp[(4) - (5)].ll_node),LLNULL,LLNULL);;}
    break;

  case 966:
#line 6894 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 967:
#line 6896 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 968:
#line 6900 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 969:
#line 6902 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (4)].ll_node); (yyval.ll_node)->entry.Template.ll_ptr1 = (yyvsp[(3) - (4)].ll_node);;}
    break;

  case 970:
#line 6911 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,HPF_INDEPENDENT_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 971:
#line 6913 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,HPF_INDEPENDENT_DIR,SMNULL, (yyvsp[(3) - (3)].ll_node), LLNULL, LLNULL);;}
    break;

  case 972:
#line 6915 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,HPF_INDEPENDENT_DIR,SMNULL, LLNULL, (yyvsp[(3) - (3)].ll_node), LLNULL);;}
    break;

  case 973:
#line 6917 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,HPF_INDEPENDENT_DIR,SMNULL, (yyvsp[(3) - (4)].ll_node), (yyvsp[(4) - (4)].ll_node),LLNULL);;}
    break;

  case 974:
#line 6953 "gram1.y"
    {(yyval.ll_node) = make_llnd(fi,REDUCTION_OP,(yyvsp[(5) - (6)].ll_node),LLNULL,SMNULL);;}
    break;

  case 975:
#line 6957 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_ASYNCHRONOUS_DIR,SMNULL,(yyvsp[(3) - (3)].ll_node),LLNULL,LLNULL);;}
    break;

  case 976:
#line 6961 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_ENDASYNCHRONOUS_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 977:
#line 6965 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_ASYNCWAIT_DIR,SMNULL,(yyvsp[(3) - (3)].ll_node),LLNULL,LLNULL);;}
    break;

  case 978:
#line 6969 "gram1.y"
    {  
            if(((yyval.symbol)=(yyvsp[(1) - (1)].hash_entry)->id_attr) == SMNULL) {
                errstr("'%s' is not declared as ASYNCID", (yyvsp[(1) - (1)].hash_entry)->ident, 115);
                (yyval.symbol) = make_local_entity((yyvsp[(1) - (1)].hash_entry),ASYNC_ID,global_default,LOCAL);
            } else {
              if((yyval.symbol)->variant != ASYNC_ID)
                errstr("'%s' is not declared as ASYNCID", (yyvsp[(1) - (1)].hash_entry)->ident, 115);
            }
     ;}
    break;

  case 979:
#line 6981 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,VAR_REF, LLNULL, LLNULL, (yyvsp[(1) - (1)].symbol));;}
    break;

  case 980:
#line 6983 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ARRAY_REF, (yyvsp[(3) - (4)].ll_node), LLNULL, (yyvsp[(1) - (4)].symbol));;}
    break;

  case 981:
#line 6987 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_F90_DIR,SMNULL,(yyvsp[(3) - (5)].ll_node),(yyvsp[(5) - (5)].ll_node),LLNULL);;}
    break;

  case 982:
#line 6990 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_DEBUG_DIR,SMNULL,(yyvsp[(3) - (3)].ll_node),LLNULL,LLNULL);;}
    break;

  case 983:
#line 6992 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_DEBUG_DIR,SMNULL,(yyvsp[(3) - (6)].ll_node),(yyvsp[(5) - (6)].ll_node),LLNULL);;}
    break;

  case 984:
#line 6996 "gram1.y"
    { 
              (yyval.ll_node) = set_ll_list((yyvsp[(2) - (2)].ll_node), LLNULL, EXPR_LIST);
              endioctl();
            ;}
    break;

  case 985:
#line 7001 "gram1.y"
    { 
              (yyval.ll_node) = set_ll_list((yyvsp[(1) - (4)].ll_node), (yyvsp[(4) - (4)].ll_node), EXPR_LIST);
              endioctl();
            ;}
    break;

  case 986:
#line 7008 "gram1.y"
    { (yyval.ll_node)  = make_llnd(fi, KEYWORD_ARG, (yyvsp[(1) - (2)].ll_node), (yyvsp[(2) - (2)].ll_node), SMNULL); ;}
    break;

  case 987:
#line 7011 "gram1.y"
    {
	         (yyval.ll_node) = make_llnd(fi,INT_VAL, LLNULL, LLNULL, SMNULL);
	         (yyval.ll_node)->entry.ival = atoi(yytext);
	         (yyval.ll_node)->type = global_int;
	        ;}
    break;

  case 988:
#line 7019 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_ENDDEBUG_DIR,SMNULL,(yyvsp[(3) - (3)].ll_node),LLNULL,LLNULL);;}
    break;

  case 989:
#line 7023 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_INTERVAL_DIR,SMNULL,(yyvsp[(3) - (3)].ll_node),LLNULL,LLNULL);;}
    break;

  case 990:
#line 7027 "gram1.y"
    { (yyval.ll_node) = LLNULL;;}
    break;

  case 991:
#line 7030 "gram1.y"
    { if((yyvsp[(1) - (1)].ll_node)->type->variant != T_INT)             
                    err("Illegal interval number", 78);
                  (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);
                 ;}
    break;

  case 992:
#line 7038 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_EXIT_INTERVAL_DIR,SMNULL,(yyvsp[(3) - (3)].ll_node),LLNULL,LLNULL);;}
    break;

  case 993:
#line 7042 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_ENDINTERVAL_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 994:
#line 7046 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_TRACEON_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 995:
#line 7050 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_TRACEOFF_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 996:
#line 7054 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_BARRIER_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 997:
#line 7058 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_CHECK_DIR,SMNULL,(yyvsp[(9) - (9)].ll_node),(yyvsp[(5) - (9)].ll_node),LLNULL); ;}
    break;

  case 998:
#line 7062 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_IO_MODE_DIR,SMNULL,(yyvsp[(4) - (5)].ll_node),LLNULL,LLNULL);;}
    break;

  case 999:
#line 7065 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 1000:
#line 7067 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 1001:
#line 7071 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_ASYNC_OP,LLNULL,LLNULL,SMNULL);;}
    break;

  case 1002:
#line 7073 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_LOCAL_OP, LLNULL,LLNULL,SMNULL);;}
    break;

  case 1003:
#line 7075 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,PARALLEL_OP, LLNULL,LLNULL,SMNULL);;}
    break;

  case 1004:
#line 7079 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_SHADOW_ADD_DIR,SMNULL,(yyvsp[(4) - (9)].ll_node),(yyvsp[(6) - (9)].ll_node),(yyvsp[(9) - (9)].ll_node)); ;}
    break;

  case 1005:
#line 7083 "gram1.y"
    {
                 if((yyvsp[(1) - (4)].ll_node)->type->variant != T_ARRAY) 
                    errstr("'%s' isn't array", (yyvsp[(1) - (4)].ll_node)->entry.Template.symbol->ident, 66);
                 if(!((yyvsp[(1) - (4)].ll_node)->entry.Template.symbol->attr & TEMPLATE_BIT))
                    errstr("'%s' isn't TEMPLATE", (yyvsp[(1) - (4)].ll_node)->entry.Template.symbol->ident, 628);
                 (yyvsp[(1) - (4)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(3) - (4)].ll_node);
                 (yyval.ll_node) = (yyvsp[(1) - (4)].ll_node);
                 /*$$->type = $1->type->entry.ar_decl.base_type;*/
               ;}
    break;

  case 1006:
#line 7095 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 1007:
#line 7097 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 1008:
#line 7101 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 1009:
#line 7103 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 1010:
#line 7107 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(2) - (2)].ll_node);;}
    break;

  case 1011:
#line 7109 "gram1.y"
    { (yyval.ll_node) = LLNULL; opt_kwd_ = NO;;}
    break;

  case 1012:
#line 7113 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_LOCALIZE_DIR,SMNULL,(yyvsp[(4) - (7)].ll_node),(yyvsp[(6) - (7)].ll_node),LLNULL); ;}
    break;

  case 1013:
#line 7117 "gram1.y"
    {
                 if((yyvsp[(1) - (1)].ll_node)->type->variant != T_ARRAY) 
                    errstr("'%s' isn't array", (yyvsp[(1) - (1)].ll_node)->entry.Template.symbol->ident, 66); 
                 (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);
                ;}
    break;

  case 1014:
#line 7123 "gram1.y"
    {
                 if((yyvsp[(1) - (4)].ll_node)->type->variant != T_ARRAY) 
                    errstr("'%s' isn't array", (yyvsp[(1) - (4)].ll_node)->entry.Template.symbol->ident, 66); 
                                 
                 (yyvsp[(1) - (4)].ll_node)->entry.Template.ll_ptr1 = (yyvsp[(3) - (4)].ll_node);
                 (yyval.ll_node) = (yyvsp[(1) - (4)].ll_node);
                 (yyval.ll_node)->type = (yyvsp[(1) - (4)].ll_node)->type->entry.ar_decl.base_type;
                ;}
    break;

  case 1015:
#line 7135 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 1016:
#line 7137 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 1017:
#line 7141 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 1018:
#line 7143 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,DDOT, LLNULL, LLNULL, SMNULL);;}
    break;

  case 1019:
#line 7147 "gram1.y"
    { 
            (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
            (yyval.ll_node)->entry.string_val = (char *) "*";
            (yyval.ll_node)->type = global_string;
          ;}
    break;

  case 1020:
#line 7155 "gram1.y"
    { 
                PTR_LLND q;
                if((yyvsp[(16) - (16)].ll_node))
                  q = make_llnd(fi,ARRAY_OP, (yyvsp[(14) - (16)].ll_node), (yyvsp[(16) - (16)].ll_node), SMNULL);
                else
                  q = (yyvsp[(14) - (16)].ll_node);                  
                (yyval.bf_node) = get_bfnd(fi,DVM_CP_CREATE_DIR,SMNULL,(yyvsp[(3) - (16)].ll_node),(yyvsp[(8) - (16)].ll_node),q); 
              ;}
    break;

  case 1021:
#line 7166 "gram1.y"
    { (yyval.ll_node) = LLNULL; ;}
    break;

  case 1022:
#line 7168 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, PARALLEL_OP, LLNULL, LLNULL, SMNULL); ;}
    break;

  case 1023:
#line 7170 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_LOCAL_OP, LLNULL, LLNULL, SMNULL); ;}
    break;

  case 1024:
#line 7174 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_CP_LOAD_DIR,SMNULL,(yyvsp[(3) - (3)].ll_node),LLNULL,LLNULL); ;}
    break;

  case 1025:
#line 7178 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_CP_SAVE_DIR,SMNULL,(yyvsp[(3) - (3)].ll_node),LLNULL,LLNULL); ;}
    break;

  case 1026:
#line 7180 "gram1.y"
    {
                PTR_LLND q;
                q = make_llnd(fi,ACC_ASYNC_OP,LLNULL,LLNULL,SMNULL);
                (yyval.bf_node) = get_bfnd(fi,DVM_CP_SAVE_DIR,SMNULL,(yyvsp[(3) - (6)].ll_node),q,LLNULL);
              ;}
    break;

  case 1027:
#line 7188 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_CP_WAIT_DIR,SMNULL,(yyvsp[(3) - (9)].ll_node),(yyvsp[(8) - (9)].ll_node),LLNULL); ;}
    break;

  case 1028:
#line 7192 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_TEMPLATE_CREATE_DIR,SMNULL,(yyvsp[(4) - (5)].ll_node),LLNULL,LLNULL); ;}
    break;

  case 1029:
#line 7195 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 1030:
#line 7197 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); ;}
    break;

  case 1031:
#line 7201 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,DVM_TEMPLATE_DELETE_DIR,SMNULL,(yyvsp[(4) - (5)].ll_node),LLNULL,LLNULL); ;}
    break;

  case 1059:
#line 7235 "gram1.y"
    {
          (yyval.bf_node) = get_bfnd(fi,OMP_ONETHREAD_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	;}
    break;

  case 1060:
#line 7241 "gram1.y"
    {
  	   (yyval.bf_node) = make_endparallel();
	;}
    break;

  case 1061:
#line 7247 "gram1.y"
    {
  	   (yyval.bf_node) = make_parallel();
           (yyval.bf_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (4)].ll_node);
	   opt_kwd_ = NO;
	;}
    break;

  case 1062:
#line 7253 "gram1.y"
    {
  	   (yyval.bf_node) = make_parallel();
	   opt_kwd_ = NO;
	;}
    break;

  case 1063:
#line 7259 "gram1.y"
    { 
		(yyval.ll_node) = set_ll_list((yyvsp[(3) - (4)].ll_node),LLNULL,EXPR_LIST);
	;}
    break;

  case 1064:
#line 7263 "gram1.y"
    { 
		(yyval.ll_node) = set_ll_list((yyvsp[(1) - (5)].ll_node),(yyvsp[(4) - (5)].ll_node),EXPR_LIST);	
	;}
    break;

  case 1074:
#line 7280 "gram1.y"
    {
		(yyval.ll_node) = (yyvsp[(4) - (5)].ll_node);
        ;}
    break;

  case 1075:
#line 7285 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,OMP_PRIVATE,(yyvsp[(2) - (2)].ll_node),LLNULL,SMNULL);
	;}
    break;

  case 1076:
#line 7290 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,OMP_FIRSTPRIVATE,(yyvsp[(2) - (2)].ll_node),LLNULL,SMNULL);
	;}
    break;

  case 1077:
#line 7296 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,OMP_LASTPRIVATE,(yyvsp[(2) - (2)].ll_node),LLNULL,SMNULL);
	;}
    break;

  case 1078:
#line 7302 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,OMP_COPYIN,(yyvsp[(2) - (2)].ll_node),LLNULL,SMNULL);
	;}
    break;

  case 1079:
#line 7308 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,OMP_SHARED,(yyvsp[(2) - (2)].ll_node),LLNULL,SMNULL);
	;}
    break;

  case 1080:
#line 7313 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,OMP_DEFAULT,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);
	;}
    break;

  case 1081:
#line 7319 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		(yyval.ll_node)->entry.string_val = (char *) "private";
		(yyval.ll_node)->type = global_string;
	;}
    break;

  case 1082:
#line 7325 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		(yyval.ll_node)->entry.string_val = (char *) "shared";
		(yyval.ll_node)->type = global_string;
	;}
    break;

  case 1083:
#line 7331 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		(yyval.ll_node)->entry.string_val = (char *) "none";
		(yyval.ll_node)->type = global_string;
	;}
    break;

  case 1084:
#line 7338 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,OMP_IF,(yyvsp[(3) - (4)].ll_node),LLNULL,SMNULL);
	;}
    break;

  case 1085:
#line 7344 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,OMP_NUM_THREADS,(yyvsp[(3) - (4)].ll_node),LLNULL,SMNULL);
	;}
    break;

  case 1086:
#line 7350 "gram1.y"
    {
		PTR_LLND q;
		q = set_ll_list((yyvsp[(3) - (4)].ll_node),LLNULL,EXPR_LIST);
		(yyval.ll_node) = make_llnd(fi,OMP_REDUCTION,q,LLNULL,SMNULL);
	;}
    break;

  case 1087:
#line 7357 "gram1.y"
    {(yyval.ll_node) = make_llnd(fi,DDOT,(yyvsp[(2) - (4)].ll_node),(yyvsp[(4) - (4)].ll_node),SMNULL);;}
    break;

  case 1089:
#line 7363 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "+";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 1090:
#line 7369 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "-";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 1091:
#line 7376 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "*";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 1092:
#line 7382 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "/";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 1093:
#line 7388 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "min";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 1094:
#line 7394 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "max";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 1095:
#line 7400 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) ".or.";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 1096:
#line 7406 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) ".and.";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 1097:
#line 7412 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) ".eqv.";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 1098:
#line 7418 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) ".neqv.";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 1099:
#line 7424 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "iand";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 1100:
#line 7430 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "ieor";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 1101:
#line 7436 "gram1.y"
    {
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "ior";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 1102:
#line 7442 "gram1.y"
    { err("Illegal reduction operation name", 70);
               errcnt--;
              (yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              (yyval.ll_node)->entry.string_val = (char *) "unknown";
              (yyval.ll_node)->type = global_string;
             ;}
    break;

  case 1103:
#line 7452 "gram1.y"
    {
  	   (yyval.bf_node) = make_sections((yyvsp[(4) - (4)].ll_node));
	   opt_kwd_ = NO;
	;}
    break;

  case 1104:
#line 7457 "gram1.y"
    {
  	   (yyval.bf_node) = make_sections(LLNULL);
	   opt_kwd_ = NO;
	;}
    break;

  case 1105:
#line 7463 "gram1.y"
    { 
		(yyval.ll_node) = set_ll_list((yyvsp[(3) - (4)].ll_node),LLNULL,EXPR_LIST);
	;}
    break;

  case 1106:
#line 7467 "gram1.y"
    { 
		(yyval.ll_node) = set_ll_list((yyvsp[(1) - (5)].ll_node),(yyvsp[(4) - (5)].ll_node),EXPR_LIST);
	;}
    break;

  case 1111:
#line 7479 "gram1.y"
    {
		PTR_LLND q;
   	        (yyval.bf_node) = make_endsections();
		q = set_ll_list((yyvsp[(4) - (4)].ll_node),LLNULL,EXPR_LIST);
                (yyval.bf_node)->entry.Template.ll_ptr1 = q;
                opt_kwd_ = NO;
	;}
    break;

  case 1112:
#line 7487 "gram1.y"
    {
   	        (yyval.bf_node) = make_endsections();
	        opt_kwd_ = NO; 
	;}
    break;

  case 1113:
#line 7493 "gram1.y"
    {
           (yyval.bf_node) = make_ompsection();
	;}
    break;

  case 1114:
#line 7499 "gram1.y"
    {
           (yyval.bf_node) = get_bfnd(fi,OMP_DO_DIR,SMNULL,(yyvsp[(4) - (4)].ll_node),LLNULL,LLNULL);
	   opt_kwd_ = NO;
	;}
    break;

  case 1115:
#line 7504 "gram1.y"
    {
           (yyval.bf_node) = get_bfnd(fi,OMP_DO_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	   opt_kwd_ = NO;
	;}
    break;

  case 1116:
#line 7510 "gram1.y"
    {
		PTR_LLND q;
		q = set_ll_list((yyvsp[(4) - (4)].ll_node),LLNULL,EXPR_LIST);
	        (yyval.bf_node) = get_bfnd(fi,OMP_END_DO_DIR,SMNULL,q,LLNULL,LLNULL);
      	        opt_kwd_ = NO;
	;}
    break;

  case 1117:
#line 7517 "gram1.y"
    {
           (yyval.bf_node) = get_bfnd(fi,OMP_END_DO_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	   opt_kwd_ = NO;
	;}
    break;

  case 1118:
#line 7523 "gram1.y"
    { 
		(yyval.ll_node) = set_ll_list((yyvsp[(3) - (4)].ll_node),LLNULL,EXPR_LIST);
	;}
    break;

  case 1119:
#line 7527 "gram1.y"
    { 
		(yyval.ll_node) = set_ll_list((yyvsp[(1) - (5)].ll_node),(yyvsp[(4) - (5)].ll_node),EXPR_LIST);
	;}
    break;

  case 1126:
#line 7541 "gram1.y"
    {
		/*$$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		$$->entry.string_val = (char *) "ORDERED";
		$$->type = global_string;*/
                (yyval.ll_node) = make_llnd(fi,OMP_ORDERED,LLNULL,LLNULL,SMNULL);
	;}
    break;

  case 1127:
#line 7550 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,OMP_SCHEDULE,(yyvsp[(4) - (7)].ll_node),(yyvsp[(6) - (7)].ll_node),SMNULL);
	;}
    break;

  case 1128:
#line 7554 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,OMP_SCHEDULE,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);
	;}
    break;

  case 1129:
#line 7560 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		(yyval.ll_node)->entry.string_val = (char *) "STATIC";
		(yyval.ll_node)->type = global_string;
		
	;}
    break;

  case 1130:
#line 7567 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		(yyval.ll_node)->entry.string_val = (char *) "DYNAMIC";
		(yyval.ll_node)->type = global_string;
		
	;}
    break;

  case 1131:
#line 7574 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		(yyval.ll_node)->entry.string_val = (char *) "GUIDED";
		(yyval.ll_node)->type = global_string;
		
	;}
    break;

  case 1132:
#line 7581 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		(yyval.ll_node)->entry.string_val = (char *) "RUNTIME";
		(yyval.ll_node)->type = global_string;
		
	;}
    break;

  case 1133:
#line 7590 "gram1.y"
    {
  	   (yyval.bf_node) = make_single();
           (yyval.bf_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (4)].ll_node);
	   opt_kwd_ = NO;
	;}
    break;

  case 1134:
#line 7596 "gram1.y"
    {
  	   (yyval.bf_node) = make_single();
	   opt_kwd_ = NO;
	;}
    break;

  case 1135:
#line 7602 "gram1.y"
    { 
		(yyval.ll_node) = set_ll_list((yyvsp[(3) - (4)].ll_node),LLNULL,EXPR_LIST);
	;}
    break;

  case 1136:
#line 7606 "gram1.y"
    { 
		(yyval.ll_node) = set_ll_list((yyvsp[(1) - (5)].ll_node),(yyvsp[(4) - (5)].ll_node),EXPR_LIST);
	;}
    break;

  case 1139:
#line 7616 "gram1.y"
    {
  	   (yyval.bf_node) = make_endsingle();
           (yyval.bf_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (4)].ll_node);
	   opt_kwd_ = NO;
	;}
    break;

  case 1140:
#line 7622 "gram1.y"
    {
  	   (yyval.bf_node) = make_endsingle();
	   opt_kwd_ = NO;
	;}
    break;

  case 1141:
#line 7628 "gram1.y"
    { 
		(yyval.ll_node) = set_ll_list((yyvsp[(3) - (4)].ll_node),LLNULL,EXPR_LIST);
	;}
    break;

  case 1142:
#line 7632 "gram1.y"
    { 
		(yyval.ll_node) = set_ll_list((yyvsp[(1) - (5)].ll_node),(yyvsp[(4) - (5)].ll_node),EXPR_LIST);
	;}
    break;

  case 1145:
#line 7643 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,OMP_COPYPRIVATE,(yyvsp[(2) - (2)].ll_node),LLNULL,SMNULL);
	;}
    break;

  case 1146:
#line 7649 "gram1.y"
    {
		(yyval.ll_node) = make_llnd(fi,OMP_NOWAIT,LLNULL,LLNULL,SMNULL);
	;}
    break;

  case 1147:
#line 7655 "gram1.y"
    {
           (yyval.bf_node) = make_workshare();
	;}
    break;

  case 1148:
#line 7660 "gram1.y"
    {
		PTR_LLND q;
   	        (yyval.bf_node) = make_endworkshare();
		q = set_ll_list((yyvsp[(4) - (4)].ll_node),LLNULL,EXPR_LIST);
                (yyval.bf_node)->entry.Template.ll_ptr1 = q;
  	        opt_kwd_ = NO;
	;}
    break;

  case 1149:
#line 7668 "gram1.y"
    {
   	        (yyval.bf_node) = make_endworkshare();
                opt_kwd_ = NO;
	;}
    break;

  case 1150:
#line 7674 "gram1.y"
    {
           (yyval.bf_node) = get_bfnd(fi,OMP_PARALLEL_DO_DIR,SMNULL,(yyvsp[(4) - (4)].ll_node),LLNULL,LLNULL);
	   opt_kwd_ = NO;
	;}
    break;

  case 1151:
#line 7679 "gram1.y"
    {
           (yyval.bf_node) = get_bfnd(fi,OMP_PARALLEL_DO_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	   opt_kwd_ = NO;
	;}
    break;

  case 1152:
#line 7686 "gram1.y"
    { 
		(yyval.ll_node) = set_ll_list((yyvsp[(3) - (4)].ll_node),LLNULL,EXPR_LIST);
	;}
    break;

  case 1153:
#line 7690 "gram1.y"
    { 
		(yyval.ll_node) = set_ll_list((yyvsp[(1) - (5)].ll_node),(yyvsp[(4) - (5)].ll_node),EXPR_LIST);
	;}
    break;

  case 1165:
#line 7710 "gram1.y"
    {
           (yyval.bf_node) = get_bfnd(fi,OMP_END_PARALLEL_DO_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	;}
    break;

  case 1166:
#line 7715 "gram1.y"
    {
           (yyval.bf_node) = make_parallelsections((yyvsp[(4) - (4)].ll_node));
	   opt_kwd_ = NO;
	;}
    break;

  case 1167:
#line 7720 "gram1.y"
    {
           (yyval.bf_node) = make_parallelsections(LLNULL);
	   opt_kwd_ = NO;
	;}
    break;

  case 1168:
#line 7727 "gram1.y"
    {
           (yyval.bf_node) = make_endparallelsections();
	;}
    break;

  case 1169:
#line 7732 "gram1.y"
    {
           (yyval.bf_node) = make_parallelworkshare();
           (yyval.bf_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (4)].ll_node);
	   opt_kwd_ = NO;
	;}
    break;

  case 1170:
#line 7738 "gram1.y"
    {
           (yyval.bf_node) = make_parallelworkshare();
	   opt_kwd_ = NO;
	;}
    break;

  case 1171:
#line 7744 "gram1.y"
    {
           (yyval.bf_node) = make_endparallelworkshare();
	;}
    break;

  case 1172:
#line 7749 "gram1.y"
    { 
	   (yyval.bf_node) = get_bfnd(fi,OMP_THREADPRIVATE_DIR, SMNULL, (yyvsp[(3) - (3)].ll_node), LLNULL, LLNULL);
	;}
    break;

  case 1173:
#line 7754 "gram1.y"
    {
  	   (yyval.bf_node) = make_master();
	;}
    break;

  case 1174:
#line 7759 "gram1.y"
    {
  	   (yyval.bf_node) = make_endmaster();
	;}
    break;

  case 1175:
#line 7763 "gram1.y"
    {
  	   (yyval.bf_node) = make_ordered();
	;}
    break;

  case 1176:
#line 7768 "gram1.y"
    {
  	   (yyval.bf_node) = make_endordered();
	;}
    break;

  case 1177:
#line 7773 "gram1.y"
    {
           (yyval.bf_node) = get_bfnd(fi,OMP_BARRIER_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	;}
    break;

  case 1178:
#line 7777 "gram1.y"
    {
           (yyval.bf_node) = get_bfnd(fi,OMP_ATOMIC_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	;}
    break;

  case 1179:
#line 7782 "gram1.y"
    {
           (yyval.bf_node) = get_bfnd(fi,OMP_FLUSH_DIR,SMNULL,(yyvsp[(3) - (3)].ll_node),LLNULL,LLNULL);
	;}
    break;

  case 1180:
#line 7786 "gram1.y"
    {
           (yyval.bf_node) = get_bfnd(fi,OMP_FLUSH_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	;}
    break;

  case 1181:
#line 7792 "gram1.y"
    {
  	   (yyval.bf_node) = make_critical();
           (yyval.bf_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (5)].ll_node);
	;}
    break;

  case 1182:
#line 7797 "gram1.y"
    {
  	   (yyval.bf_node) = make_critical();
	;}
    break;

  case 1183:
#line 7803 "gram1.y"
    {
  	   (yyval.bf_node) = make_endcritical();
           (yyval.bf_node)->entry.Template.ll_ptr1 = (yyvsp[(4) - (5)].ll_node);
	;}
    break;

  case 1184:
#line 7808 "gram1.y"
    {
  	   (yyval.bf_node) = make_endcritical();
	;}
    break;

  case 1185:
#line 7814 "gram1.y"
    { 
		PTR_SYMB s;
		PTR_LLND l;
		s = make_common((yyvsp[(2) - (5)].hash_entry)); 
		l = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
		(yyval.ll_node) = make_llnd(fi,OMP_THREADPRIVATE, l, LLNULL, SMNULL);
	;}
    break;

  case 1186:
#line 7824 "gram1.y"
    {
		(yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST);
	;}
    break;

  case 1187:
#line 7828 "gram1.y"
    {	
		(yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST);
	;}
    break;

  case 1188:
#line 7832 "gram1.y"
    { 
		(yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST);
	;}
    break;

  case 1189:
#line 7836 "gram1.y"
    { 
		(yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST);
	;}
    break;

  case 1190:
#line 7841 "gram1.y"
    {
		operator_slash = 1;
	;}
    break;

  case 1191:
#line 7844 "gram1.y"
    {
		operator_slash = 0;
	;}
    break;

  case 1199:
#line 7858 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,ACC_REGION_DIR,SMNULL,(yyvsp[(3) - (3)].ll_node),LLNULL,LLNULL);;}
    break;

  case 1200:
#line 7862 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,ACC_CHECKSECTION_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 1201:
#line 7866 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,ACC_GET_ACTUAL_DIR,SMNULL,(yyvsp[(4) - (5)].ll_node),LLNULL,LLNULL);;}
    break;

  case 1202:
#line 7868 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,ACC_GET_ACTUAL_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 1203:
#line 7870 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,ACC_GET_ACTUAL_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 1204:
#line 7874 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,ACC_ACTUAL_DIR,SMNULL,(yyvsp[(4) - (5)].ll_node),LLNULL,LLNULL);;}
    break;

  case 1205:
#line 7876 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,ACC_ACTUAL_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 1206:
#line 7878 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,ACC_ACTUAL_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 1207:
#line 7882 "gram1.y"
    { (yyval.ll_node) = LLNULL;;}
    break;

  case 1208:
#line 7884 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node); ;}
    break;

  case 1209:
#line 7888 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 1210:
#line 7890 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 1211:
#line 7894 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(2) - (2)].ll_node);;}
    break;

  case 1212:
#line 7897 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(2) - (2)].ll_node);;}
    break;

  case 1213:
#line 7900 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(2) - (2)].ll_node);;}
    break;

  case 1214:
#line 7905 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_INOUT_OP,(yyvsp[(3) - (4)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1215:
#line 7907 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_IN_OP,(yyvsp[(3) - (4)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1216:
#line 7909 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_OUT_OP,(yyvsp[(3) - (4)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1217:
#line 7911 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_LOCAL_OP,(yyvsp[(3) - (4)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1218:
#line 7913 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_INLOCAL_OP,(yyvsp[(3) - (4)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1219:
#line 7917 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_TARGETS_OP,(yyvsp[(3) - (4)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1220:
#line 7921 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_ASYNC_OP,LLNULL,LLNULL,SMNULL);;}
    break;

  case 1221:
#line 7926 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(1) - (1)].ll_node);;}
    break;

  case 1222:
#line 7930 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 1223:
#line 7932 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 1224:
#line 7936 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_HOST_OP, LLNULL,LLNULL,SMNULL);;}
    break;

  case 1225:
#line 7938 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_CUDA_OP, LLNULL,LLNULL,SMNULL);;}
    break;

  case 1226:
#line 7942 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,ACC_END_REGION_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 1227:
#line 7946 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,ACC_END_CHECKSECTION_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 1228:
#line 7950 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,ACC_ROUTINE_DIR,SMNULL,(yyvsp[(3) - (3)].ll_node),LLNULL,LLNULL);;}
    break;

  case 1229:
#line 7954 "gram1.y"
    { (yyval.ll_node) = LLNULL; ;}
    break;

  case 1230:
#line 7956 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(2) - (2)].ll_node);;}
    break;

  case 1237:
#line 7968 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,SPF_ANALYSIS_DIR,SMNULL,(yyvsp[(4) - (5)].ll_node),LLNULL,LLNULL);;}
    break;

  case 1238:
#line 7972 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,SPF_PARALLEL_DIR,SMNULL,(yyvsp[(4) - (5)].ll_node),LLNULL,LLNULL);;}
    break;

  case 1239:
#line 7976 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,SPF_TRANSFORM_DIR,SMNULL,(yyvsp[(4) - (5)].ll_node),LLNULL,LLNULL);;}
    break;

  case 1240:
#line 7980 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,SPF_PARALLEL_REG_DIR,(yyvsp[(3) - (3)].symbol),LLNULL,LLNULL,LLNULL);;}
    break;

  case 1241:
#line 7982 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,SPF_PARALLEL_REG_DIR,(yyvsp[(3) - (10)].symbol),(yyvsp[(8) - (10)].ll_node),(yyvsp[(10) - (10)].ll_node),LLNULL);;}
    break;

  case 1242:
#line 7984 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,SPF_PARALLEL_REG_DIR,(yyvsp[(3) - (10)].symbol),(yyvsp[(10) - (10)].ll_node),(yyvsp[(8) - (10)].ll_node),LLNULL);;}
    break;

  case 1243:
#line 7988 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 1244:
#line 7990 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 1245:
#line 7994 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_CODE_COVERAGE_OP,LLNULL,LLNULL,SMNULL);;}
    break;

  case 1246:
#line 7998 "gram1.y"
    { (yyval.ll_node) = LLNULL;;}
    break;

  case 1247:
#line 8000 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(5) - (6)].ll_node);;}
    break;

  case 1248:
#line 8004 "gram1.y"
    { (yyval.ll_node) = LLNULL;;}
    break;

  case 1249:
#line 8006 "gram1.y"
    { (yyval.ll_node) = (yyvsp[(5) - (6)].ll_node);;}
    break;

  case 1250:
#line 8010 "gram1.y"
    { (yyval.bf_node) = get_bfnd(fi,SPF_END_PARALLEL_REG_DIR,SMNULL,LLNULL,LLNULL,LLNULL);;}
    break;

  case 1251:
#line 8014 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 1252:
#line 8016 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 1258:
#line 8027 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,REDUCTION_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL); ;}
    break;

  case 1259:
#line 8031 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_PRIVATE_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1260:
#line 8035 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_PROCESS_PRIVATE_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1261:
#line 8039 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_COVER_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1262:
#line 8043 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_PARAMETER_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1263:
#line 8046 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 1264:
#line 8048 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); ;}
    break;

  case 1265:
#line 8052 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi, ASSGN_OP, (yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), SMNULL); ;}
    break;

  case 1266:
#line 8056 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 1267:
#line 8058 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 1271:
#line 8067 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SHADOW_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1272:
#line 8071 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACROSS_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1273:
#line 8075 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,REMOTE_ACCESS_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1274:
#line 8079 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 1275:
#line 8081 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 1276:
#line 8085 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_NOINLINE_OP,LLNULL,LLNULL,SMNULL);;}
    break;

  case 1277:
#line 8087 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_FISSION_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1278:
#line 8089 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_EXPAND_OP,LLNULL,LLNULL,SMNULL);;}
    break;

  case 1279:
#line 8091 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_EXPAND_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1280:
#line 8094 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_SHRINK_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1281:
#line 8096 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_UNROLL_OP,LLNULL,LLNULL,SMNULL);;}
    break;

  case 1282:
#line 8098 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_UNROLL_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1283:
#line 8100 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_MERGE_OP,LLNULL,LLNULL,SMNULL);;}
    break;

  case 1284:
#line 8104 "gram1.y"
    {
               (yyval.ll_node) = set_ll_list((yyvsp[(1) - (5)].ll_node), (yyvsp[(3) - (5)].ll_node), EXPR_LIST);
               (yyval.ll_node) = set_ll_list((yyval.ll_node), (yyvsp[(5) - (5)].ll_node), EXPR_LIST);
             ;}
    break;

  case 1285:
#line 8111 "gram1.y"
    { (yyval.symbol) = make_parallel_region((yyvsp[(1) - (1)].hash_entry));;}
    break;

  case 1286:
#line 8115 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node), LLNULL, EXPR_LIST); ;}
    break;

  case 1287:
#line 8117 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node), (yyvsp[(3) - (3)].ll_node), EXPR_LIST); ;}
    break;

  case 1288:
#line 8121 "gram1.y"
    {  (yyval.bf_node) = get_bfnd(fi,SPF_CHECKPOINT_DIR,SMNULL,(yyvsp[(4) - (5)].ll_node),LLNULL,LLNULL);;}
    break;

  case 1289:
#line 8125 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 1290:
#line 8127 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 1291:
#line 8131 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_TYPE_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1292:
#line 8133 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_VARLIST_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1293:
#line 8135 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_EXCEPT_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1294:
#line 8137 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_FILES_COUNT_OP,(yyvsp[(4) - (5)].ll_node),LLNULL,SMNULL);;}
    break;

  case 1295:
#line 8139 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_INTERVAL_OP,(yyvsp[(4) - (7)].ll_node),(yyvsp[(6) - (7)].ll_node),SMNULL);;}
    break;

  case 1296:
#line 8143 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (1)].ll_node),LLNULL,EXPR_LIST); ;}
    break;

  case 1297:
#line 8145 "gram1.y"
    { (yyval.ll_node) = set_ll_list((yyvsp[(1) - (3)].ll_node),(yyvsp[(3) - (3)].ll_node),EXPR_LIST); ;}
    break;

  case 1298:
#line 8149 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,ACC_ASYNC_OP, LLNULL,LLNULL,SMNULL);;}
    break;

  case 1299:
#line 8151 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_FLEXIBLE_OP, LLNULL,LLNULL,SMNULL);;}
    break;

  case 1300:
#line 8155 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_TIME_OP, LLNULL,LLNULL,SMNULL);;}
    break;

  case 1301:
#line 8157 "gram1.y"
    { (yyval.ll_node) = make_llnd(fi,SPF_ITER_OP, LLNULL,LLNULL,SMNULL);;}
    break;

  case 1302:
#line 8161 "gram1.y"
    { if(position==IN_OUTSIDE)
                 err("Misplaced SPF-directive",103);
             ;}
    break;


/* Line 1267 of yacc.c.  */
#line 14174 "gram1.tab.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



