%token PERCENT 1
%token AMPERSAND 2
%token ASTER 3
%token CLUSTER 4
%token COLON 5
%token COMMA 6
%token DASTER 7
%token DEFINED_OPERATOR 8
%token DOT 9
%token DQUOTE 10
%token GLOBAL_A 11
%token LEFTAB 12
%token LEFTPAR 13
%token MINUS 14
%token PLUS 15
%token POINT_TO 16
%token QUOTE 17
%token RIGHTAB 18
%token RIGHTPAR 19
%token AND 20
%token DSLASH 21
%token EQV 22
%token EQ 23
%token EQUAL 24
%token FFALSE 25
%token GE 26
%token GT 27
%token LE 28
%token LT 29
%token NE 30
%token NEQV 31
%token NOT 32
%token OR 33
%token TTRUE 34
%token SLASH 35
%token XOR 36
%token REFERENCE 37
%token AT 38
%token ACROSS 39
%token ALIGN_WITH 40
%token ALIGN 41
%token ALLOCATABLE 42
%token ALLOCATE 43
%token ARITHIF 44
%token ASSIGNMENT 45
%token ASSIGN 46
%token ASSIGNGOTO 47
%token ASYNCHRONOUS 48
%token ASYNCID 49
%token ASYNCWAIT 50
%token BACKSPACE 51
%token BAD_CCONST 52
%token BAD_SYMBOL 53
%token BARRIER 54
%token BLOCKDATA 55
%token BLOCK 56
%token BOZ_CONSTANT 57
%token BYTE 58
%token CALL 59
%token CASE 60
%token CHARACTER 61
%token CHAR_CONSTANT 62
%token CHECK 63
%token CLOSE 64
%token COMMON 65
%token COMPLEX 66
%token COMPGOTO 67
%token CONSISTENT_GROUP 68
%token CONSISTENT_SPEC 69
%token CONSISTENT_START 70
%token CONSISTENT_WAIT 71
%token CONSISTENT 72
%token CONSTRUCT_ID 73
%token CONTAINS 74
%token CONTINUE 75
%token CORNER 76
%token CYCLE 77
%token DATA 78
%token DEALLOCATE 79
%token HPF_TEMPLATE 80
%token DEBUG 81
%token DEFAULT_CASE 82
%token DEFINE 83
%token DERIVED 84
%token DIMENSION 85
%token DISTRIBUTE 86
%token DOWHILE 87
%token DOUBLEPRECISION 88
%token DOUBLECOMPLEX 89
%token DP_CONSTANT 90
%token DVM_POINTER 91
%token DYNAMIC 92
%token ELEMENTAL 93
%token ELSE 94
%token ELSEIF 95
%token ELSEWHERE 96
%token ENDASYNCHRONOUS 97
%token ENDDEBUG 98
%token ENDINTERVAL 99
%token ENDUNIT 100
%token ENDDO 101
%token ENDFILE 102
%token ENDFORALL 103
%token ENDIF 104
%token ENDINTERFACE 105
%token ENDMODULE 106
%token ENDON 107
%token ENDSELECT 108
%token ENDTASK_REGION 109
%token ENDTYPE 110
%token ENDWHERE 111
%token ENTRY 112
%token EXIT 113
%token EOLN 114
%token EQUIVALENCE 115
%token ERROR 116
%token EXTERNAL 117
%token F90 118
%token FIND 119
%token FORALL 120
%token FORMAT 121
%token FUNCTION 122
%token GATE 123
%token GEN_BLOCK 124
%token HEAP 125
%token HIGH 126
%token IDENTIFIER 127
%token IMPLICIT 128
%token IMPLICITNONE 129
%token INCLUDE_TO 130
%token INCLUDE 131
%token INDEPENDENT 132
%token INDIRECT_ACCESS 133
%token INDIRECT_GROUP 134
%token INDIRECT 135
%token INHERIT 136
%token INQUIRE 137
%token INTERFACEASSIGNMENT 138
%token INTERFACEOPERATOR 139
%token INTERFACE 140
%token INTRINSIC 141
%token INTEGER 142
%token INTENT 143
%token INTERVAL 144
%token INOUT 145
%token IN 146
%token INT_CONSTANT 147
%token LABEL 148
%token LABEL_DECLARE 149
%token LET 150
%token LOCALIZE 151
%token LOGICAL 152
%token LOGICALIF 153
%token LOOP 154
%token LOW 155
%token MAXLOC 156
%token MAX 157
%token MAP 158
%token MINLOC 159
%token MIN 160
%token MODULE_PROCEDURE 161
%token MODULE 162
%token MULT_BLOCK 163
%token NAMEEQ 164
%token NAMELIST 165
%token NEW_VALUE 166
%token NEW 167
%token NULLIFY 168
%token OCTAL_CONSTANT 169
%token ONLY 170
%token ON 171
%token ON_DIR 172
%token ONTO 173
%token OPEN 174
%token OPERATOR 175
%token OPTIONAL 176
%token OTHERWISE 177
%token OUT 178
%token OWN 179
%token PARALLEL 180
%token PARAMETER 181
%token PAUSE 182
%token PLAINDO 183
%token PLAINGOTO 184
%token POINTER 185
%token POINTERLET 186
%token PREFETCH 187
%token PRINT 188
%token PRIVATE 189
%token PRODUCT 190
%token PROGRAM 191
%token PUBLIC 192
%token PURE 193
%token RANGE 194
%token READ 195
%token REALIGN_WITH 196
%token REALIGN 197
%token REAL 198
%token REAL_CONSTANT 199
%token RECURSIVE 200
%token REDISTRIBUTE_NEW 201
%token REDISTRIBUTE 202
%token REDUCTION_GROUP 203
%token REDUCTION_START 204
%token REDUCTION_WAIT 205
%token REDUCTION 206
%token REMOTE_ACCESS_SPEC 207
%token REMOTE_ACCESS 208
%token REMOTE_GROUP 209
%token RESET 210
%token RESULT 211
%token RETURN 212
%token REWIND 213
%token SAVE 214
%token SECTION 215
%token SELECT 216
%token SEQUENCE 217
%token SHADOW_ADD 218
%token SHADOW_COMPUTE 219
%token SHADOW_GROUP 220
%token SHADOW_RENEW 221
%token SHADOW_START_SPEC 222
%token SHADOW_START 223
%token SHADOW_WAIT_SPEC 224
%token SHADOW_WAIT 225
%token SHADOW 226
%token STAGE 227
%token STATIC 228
%token STAT 229
%token STOP 230
%token SUBROUTINE 231
%token SUM 232
%token SYNC 233
%token TARGET 234
%token TASK 235
%token TASK_REGION 236
%token THEN 237
%token TO 238
%token TRACEON 239
%token TRACEOFF 240
%token TRUNC 241
%token TYPE 242
%token TYPE_DECL 243
%token UNDER 244
%token UNKNOWN 245
%token USE 246
%token VIRTUAL 247
%token VARIABLE 248
%token WAIT 249
%token WHERE 250
%token WHERE_ASSIGN 251
%token WHILE 252
%token WITH 253
%token WRITE 254
%token COMMENT 255
%token WGT_BLOCK 256
%token HPF_PROCESSORS 257
%token IOSTAT 258
%token ERR 259
%token END 260
%token OMPDVM_ATOMIC 261
%token OMPDVM_BARRIER 262
%token OMPDVM_COPYIN 263
%token OMPDVM_COPYPRIVATE 264
%token OMPDVM_CRITICAL 265
%token OMPDVM_ONETHREAD 266
%token OMPDVM_DO 267
%token OMPDVM_DYNAMIC 268
%token OMPDVM_ENDCRITICAL 269
%token OMPDVM_ENDDO 270
%token OMPDVM_ENDMASTER 271
%token OMPDVM_ENDORDERED 272
%token OMPDVM_ENDPARALLEL 273
%token OMPDVM_ENDPARALLELDO 274
%token OMPDVM_ENDPARALLELSECTIONS 275
%token OMPDVM_ENDPARALLELWORKSHARE 276
%token OMPDVM_ENDSECTIONS 277
%token OMPDVM_ENDSINGLE 278
%token OMPDVM_ENDWORKSHARE 279
%token OMPDVM_FIRSTPRIVATE 280
%token OMPDVM_FLUSH 281
%token OMPDVM_GUIDED 282
%token OMPDVM_LASTPRIVATE 283
%token OMPDVM_MASTER 284
%token OMPDVM_NOWAIT 285
%token OMPDVM_NONE 286
%token OMPDVM_NUM_THREADS 287
%token OMPDVM_ORDERED 288
%token OMPDVM_PARALLEL 289
%token OMPDVM_PARALLELDO 290
%token OMPDVM_PARALLELSECTIONS 291
%token OMPDVM_PARALLELWORKSHARE 292
%token OMPDVM_RUNTIME 293
%token OMPDVM_SECTION 294
%token OMPDVM_SECTIONS 295
%token OMPDVM_SCHEDULE 296
%token OMPDVM_SHARED 297
%token OMPDVM_SINGLE 298
%token OMPDVM_THREADPRIVATE 299
%token OMPDVM_WORKSHARE 300
%token OMPDVM_NODES 301
%token OMPDVM_IF 302
%token IAND 303
%token IEOR 304
%token IOR 305
%token ACC_REGION 306
%token ACC_END_REGION 307
%token ACC_CHECKSECTION 308
%token ACC_END_CHECKSECTION 309
%token ACC_GET_ACTUAL 310
%token ACC_ACTUAL 311
%token ACC_TARGETS 312
%token ACC_ASYNC 313
%token ACC_HOST 314
%token ACC_CUDA 315
%token ACC_LOCAL 316
%token ACC_INLOCAL 317
%token ACC_CUDA_BLOCK 318
%token ACC_ROUTINE 319
%token ACC_TIE 320
%token BY 321
%token IO_MODE 322
%token CP_CREATE 323
%token CP_LOAD 324
%token CP_SAVE 325
%token CP_WAIT 326
%token FILES 327
%token VARLIST 328
%token STATUS 329
%token EXITINTERVAL 330
%token TEMPLATE_CREATE 331
%token TEMPLATE_DELETE 332
%token SPF_ANALYSIS 333
%token SPF_PARALLEL 334
%token SPF_TRANSFORM 335
%token SPF_NOINLINE 336
%token SPF_PARALLEL_REG 337
%token SPF_END_PARALLEL_REG 338
%token SPF_EXPAND 339
%token SPF_FISSION 340
%token SPF_SHRINK 341
%token SPF_CHECKPOINT 342
%token SPF_EXCEPT 343
%token SPF_FILES_COUNT 344
%token SPF_INTERVAL 345
%token SPF_TIME 346
%token SPF_ITER 347
%token SPF_FLEXIBLE 348
%token SPF_APPLY_REGION 349
%token SPF_APPLY_FRAGMENT 350
%token SPF_CODE_COVERAGE 351
%token SPF_UNROLL 352
%token SPF_MERGE 353
%token SPF_COVER 354
%token SPF_PROCESS_PRIVATE 355

%{
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
%}

%union {
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

/*
 * gram.head
 *
 * First part of the Fortran grammar
 *
 */

/* Specify precedences and associativities. */


%left COMMA
%nonassoc COLON
%right EQUAL
%left DEFINED_OPERATOR
%left BINARY_OP
%left EQV NEQV
%left OR XOR
%left AND
%left NOT
%nonassoc LT GT LE GE EQ NE
%left DSLASH
%left PLUS MINUS
%left ASTER SLASH
%right DASTER
%nonassoc UNARY_OP

%start program
%type <token> addop  stop att_type
%type <charv> letter
%type <charp> filename
%type <hash_entry> name opt_unit_name
%type <symbol> progname blokname args arg arglist call comblock namelist_group
%type <bf_node> program stat spec exec iffable goto logif
%type <bf_node> dcl implicit data common dimension external intrinsic attrib
%type <bf_node> equivalence namelist type_dcl end_type static
%type <bf_node> intent optional public private sequence allocatable pointer target 
%type <ll_node> implist impitem
%type <ll_node> paramlist paramitem
%type <ll_node> dim dims dimlist ubound funarglist funarg funargs opt_expr var
%type <ll_node> labellist expr uexpr lhs simple_const lengspec substring 
%type <ll_node> complex_const vec triplet
%type <ll_node> dospec  use_name_list 
%type <symbol>  do_var
%type <label>   dotarget
%type <symbol> funcname typedfunc procname
%type <ll_node> equivset equivlist
%type <ll_node> savelist saveitem

/* FORTRAN 90 */
%type <hash_entry> defined_op intrinsic_op operator construct_name
%type <bf_node> case whereable interface use_stat forall
%type <ll_node> forall_list forall_expr opt_forall_cond
%type <bf_node> module_proc_stmt do_while plain_do
%type <ll_node> opt_result_clause case_selector case_value_range
%type <ll_node> case_value_range_list proper_lengspec
%type <symbol>  opt_construct_name  
%type <symbol>  module_name 
%type <ll_node> selector initial_value clause opt_while
%type <ll_node> options attr_spec_list attr_spec intent_spec access_spec
%type <ll_node> string_constant structure_component opt_substring
%type <ll_node> array_ele_substring_func_ref ident array_element 
%type <ll_node> subscript_list asubstring equi_object
%type <ll_node> allocation_list allocate_object_list
%type <ll_node> allocate_object pointer_name_list rename_list rename_name use_name
%type <ll_node> only_list only_name proc_name_list construct_name_colon
%type <ll_node> kind numeric_bool_const integer_constant proc_attr

/*
 * used by I/O statement
 */

%type <bf_node> io read write print iofmove iofctl fmkwd ctlkwd inquire
%type <ll_node> fexpr unpar_fexpr 
%type <ll_node> ioctl ctllist ioclause nameeq inlist inelt
%type <ll_node> outlist out2 other infmt

%type <ll_node> label letgroups letgroup let
%type <ll_node> callarglist callarg
%type <data_type> typename typespec type type_implicit
%type <label> thislabel


%type <ll_node> dataimplieddo dlist dataelt datasubs datarange
%type <ll_node> iconexprlist opticonexpr iconexpr iconterm
%type <ll_node> iconfactor iconprimary
%type <symbol>  dataname d_name

/* 
 *
 * used by HPF and FDVM 
 */
%type <bf_node> dvm_specification dvm_combined_dir dvm_pointer dvm_heap
%type <bf_node> dvm_template dvm_processors dvm_indirect_group dvm_remote_group
%type <bf_node> dvm_task dvm_inherit dvm_new_value
%type <bf_node> dvm_dynamic dvm_align dvm_realign align_directive_stuff
%type <bf_node> dvm_distribute dvm_redistribute dvm_exec
%type <bf_node> dvm_parallel_on dvm_remote_access 
%type <bf_node> dvm_shadow_group dvm_shadow_start dvm_shadow_wait dvm_shadow
%type <bf_node> dvm_reduction_group dvm_reduction_start dvm_reduction_wait
%type <bf_node> dvm_task_region  dvm_end_task_region dvm_map dvm_on dvm_end_on
%type <bf_node> dvm_reset dvm_prefetch dvm_indirect_access hpf_independent 
%type <bf_node> dvm_debug_dir dvm_enddebug_dir dvm_traceon_dir dvm_traceoff_dir
%type <bf_node> dvm_interval_dir dvm_endinterval_dir dvm_exit_interval_dir dvm_barrier_dir dvm_check 
%type <bf_node> dvm_io_mode_dir dvm_shadow_add dvm_localize
%type <bf_node> dvm_cp_create dvm_cp_load dvm_cp_save dvm_cp_wait dvm_template_create dvm_template_delete
%type <bf_node> dvm_asyncid dvm_f90 dvm_asynchronous dvm_endasynchronous dvm_asyncwait
%type <bf_node> dvm_consistent_group dvm_consistent_start dvm_consistent_wait dvm_consistent
%type <ll_node> dist_name dist_name_list dist_format dist_format_list
%type <ll_node> distributee pointer_ar_elem
%type <ll_node> dist_format_clause opt_dist_format_clause  shadow_width 
%type <ll_node> opt_spec  dvm_attribute_list dvm_attribute
%type <ll_node> new_spec reduction_spec shadow_spec remote_access_spec
%type <ll_node> spec_list par_spec indirect_access_spec across_spec stage_spec
%type <ll_node> in_out_across opt_in_out
%type <ll_node> dependent_array_list dependent_array dependence  dependence_list
%type <ll_node> variable_list reduction_list shadow_list distribute_cycles
%type <ll_node> reduction reduction_op loc_op array_ident array_ident_list shadow
%type <ll_node> dyn_array_name_list dyn_array_name 
%type <ll_node> heap_array_name_list heap_array_name 
%type <ll_node> align_base align_subscript_list par_subscript_list par_subscript
%type <ll_node> remote_data_list remote_data remote_index_list remote_index
%type <ll_node> dim_ident_list dim_ident align_subscript 
%type <ll_node> realignee_list alignee realignee
%type <ll_node> dummy_array_name_list dummy_array_name 
%type <ll_node> ident_list sh_array_name 
%type <ll_node> shadow_attr_stuff sh_width  sh_width_list
%type <ll_node> pointer_var pointer_var_list dimension_list
%type <ll_node> interval_number fragment_number
%type <ll_node> task task_array opt_private_spec async 
%type <ll_node> indirect_list indirect_reference opt_onto opt_on hpf_reduction_spec
%type <ll_node> debparamlist debparam async_id_list async_id high_section
%type <ll_node> section_spec_list section_spec section ar_section low_section
%type <ll_node> consistent_spec consistent_array_name_list consistent_array_name
%type <ll_node> mode_list mode_spec opt_mode
%type <ll_node> derived_spec derived_elem derived_elem_list target_spec 
%type <ll_node> derived_subscript derived_subscript_list opt_plus_shadow plus_shadow shadow_id
%type <ll_node> template_ref template_obj shadow_axis shadow_axis_list opt_include_to 
%type <ll_node> localize_target target_subscript target_subscript_list aster_expr dummy_ident 
%type <ll_node> template_list tie_spec tied_array_list
%type <symbol> processors_name align_base_name 
%type <symbol> shadow_group_name reduction_group_name reduction_group  indirect_group_name task_name
%type <symbol> remote_group_name group_name array_name async_ident consistent_group_name consistent_group
%type <symbol> derived_target 

/* FORTRAN OPENMPDVM*/
%type <bf_node> omp_specification_directive omp_threadprivate_directive
%type <bf_node> omp_execution_directive omp_section_directive
%type <bf_node> omp_parallel_begin_directive omp_parallel_end_directive
%type <bf_node> omp_sections_begin_directive omp_sections_end_directive
%type <bf_node> omp_do_begin_directive omp_do_end_directive
%type <bf_node> omp_single_begin_directive omp_single_end_directive
%type <bf_node> omp_workshare_begin_directive omp_workshare_end_directive
%type <bf_node> omp_parallel_do_begin_directive omp_parallel_do_end_directive
%type <bf_node> omp_parallel_sections_begin_directive omp_parallel_sections_end_directive
%type <bf_node> omp_parallel_workshare_begin_directive omp_parallel_workshare_end_directive
%type <bf_node> omp_master_begin_directive omp_master_end_directive
%type <bf_node> omp_ordered_begin_directive omp_ordered_end_directive
%type <bf_node> omp_barrier_directive omp_atomic_directive omp_flush_directive
%type <bf_node> omp_critical_begin_directive omp_critical_end_directive ompdvm_onethread

%type <ll_node> parallel_clause_list parallel_clause
%type <ll_node> ompprivate_clause ompfirstprivate_clause
%type <ll_node> omplastprivate_clause ompcopyin_clause
%type <ll_node> ompshared_clause ompdefault_clause
%type <ll_node> def_expr ompif_clause ompnumthreads_clause
%type <ll_node> ompreduction_clause ompreduction ompreduction_vars
%type <ll_node> ompreduction_op
%type <ll_node> sections_clause_list sections_clause
%type <ll_node> do_clause_list do_clause omp_variable_list
%type <ll_node> omp_common_var omp_variable_list_in_par
%type <ll_node> ompordered_clause ompschedule_clause ompschedule_op
%type <ll_node> single_clause_list single_clause
%type <ll_node> end_single_clause_list end_single_clause
%type <ll_node> ompcopyprivate_clause ompnowait_clause
%type <ll_node> paralleldo_clause_list paralleldo_clause

/* FORTRAN ACC */
%type <bf_node> acc_directive acc_region acc_end_region acc_checksection acc_end_checksection
%type <bf_node> acc_get_actual acc_actual acc_routine
%type <ll_node> opt_clause acc_clause_list acc_clause data_clause async_clause targets_clause
%type <ll_node> acc_var_list computer_list computer opt_targets_clause

/* new clauses for PARALLEL directive */
%type <ll_node> private_spec cuda_block_spec sizelist 

/* SAPFOR */
%type <bf_node> spf_directive spf_analysis spf_parallel spf_transform spf_parallel_reg spf_end_parallel_reg
%type <bf_node> spf_checkpoint
%type <ll_node> analysis_spec_list analysis_spec analysis_reduction_spec analysis_private_spec analysis_parameter_spec
%type <ll_node> analysis_cover_spec analysis_process_private_spec
%type <ll_node> parallel_spec_list parallel_spec parallel_shadow_spec parallel_across_spec parallel_remote_access_spec
%type <ll_node> transform_spec_list transform_spec array_element_list spf_parameter_list spf_parameter
%type <ll_node> characteristic characteristic_list opt_clause_apply_region opt_clause_apply_fragment 
%type <ll_node> checkpoint_spec checkpoint_spec_list spf_type_list spf_type interval_spec unroll_list
%type <symbol>  region_name 

%{
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

%}
%%

program:    { $$ = BFNULL; }
	| program stat EOLN
	    { $$ = set_stat_list($1,$2); }
	;

stat:	  thislabel  entry cmnt
	    { lastwasbranch = NO;  $$ = BFNULL; }
	| thislabel  spec cmnt
            {
	       if ($2 != BFNULL) 
               {	    
	          $2->label = $1;
	          $$ = $2;
	 	  if (is_openmp_stmt) {            /*OMP*/
			is_openmp_stmt = 0;
			if($2) {                        /*OMP*/
				if ($2->decl_specs != -BIT_OPENMP) $2->decl_specs = BIT_OPENMP; /*OMP*/
			}                               /*OMP*/
		  }                                       /*OMP*/
               }
	    }
	| thislabel exec cmnt
	    { PTR_BFND p;

	     if(lastwasbranch && ! thislabel)
               /*if (warn_all)
		 warn("statement cannot be reached", 36);*/
	     lastwasbranch = thiswasbranch;
	     thiswasbranch = NO;
	     if($2) $2->label = $1;
	     if($1 && $2) $1->statbody = $2; /*8.11.06 podd*/
	     if($1) {
		/*$1->statbody = $2;*/ /*8.11.06 podd*/
		if($1->labtype == LABFORMAT)
		  err("label already that of a format",39);
		else
		  $1->labtype = LABEXEC;
	     }
	     if (is_openmp_stmt) {            /*OMP*/
			is_openmp_stmt = 0;
			if($2) {                        /*OMP*/
				if ($2->decl_specs != -BIT_OPENMP) $2->decl_specs = BIT_OPENMP; /*OMP*/
			}                               /*OMP*/
	     }                                       /*OMP*/
             for (p = pred_bfnd; $1 && 
		  ((p->variant == FOR_NODE)||(p->variant == WHILE_NODE)) &&
                  (p->entry.for_node.doend) &&
		  (p->entry.for_node.doend->stateno == $1->stateno);
		  p = p->control_parent)
                ++end_group;
	     $$ = $2;
     }
	| thislabel INCLUDE filename
		{ /* PTR_LLND p; */
			doinclude( $3 );
/*			p = make_llnd(fi, STRING_VAL, LLNULL, LLNULL, SMNULL);
			p->entry.string_val = $3;
			p->type = global_string;
			$$ = get_bfnd(fi, INCLUDE_STAT, SMNULL, p, LLNULL); */
			$$ = BFNULL;
		}
	| thislabel UNKNOWN
	    {
	      err("Unclassifiable statement", 10);
	      flline();
	      $$ = BFNULL;
	    };
	| COMMENT
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
 	      $$ = BFNULL;         
            }

	| error
	    { 
	      flline();	 needkwd = NO;	inioctl = NO;
/*!!!*/
              opt_kwd_ = NO; intonly = NO; opt_kwd_hedr = NO; opt_kwd_r = NO; as_op_kwd_= NO; optcorner = NO;
	      yyerrok; yyclearin;  $$ = BFNULL;
	    }
	;

thislabel:  LABEL
	    {
	    if(yystno)
	      {
	      $$ = thislabel =	make_label_node(fi,yystno);
	      thislabel->scope = cur_scope();
	      if (thislabel->labdefined && (thislabel->scope == cur_scope()))
		 errstr("Label %s already defined",convic(thislabel->stateno),40);
	      else
		 thislabel->labdefined = YES;
	      }
	    else
	      $$ = thislabel = LBNULL;
	    }
	;

entry:	  PROGRAM new_prog progname
	    { PTR_BFND p;

	        if (pred_bfnd != global_bfnd)
		    err("Misplaced PROGRAM statement", 33);
		p = get_bfnd(fi,PROG_HEDR, $3, LLNULL, LLNULL, LLNULL);
		$3->entry.prog_decl.prog_hedr=p;
 		set_blobs(p, global_bfnd, NEW_GROUP1);
	        add_scope_level(p, NO);
	        position = IN_PROC;
	    }

	| BLOCKDATA new_prog blokname
	    {  PTR_BFND q = BFNULL;

	      $3->variant = PROCEDURE_NAME;
	      $3->decl = YES;   /* variable declaration has been seen. */
	      q = get_bfnd(fi,BLOCK_DATA, $3, LLNULL, LLNULL, LLNULL);
	      set_blobs(q, global_bfnd, NEW_GROUP1);
              add_scope_level(q, NO);
	    }

	| SUBROUTINE new_prog procname arglist
	    { 
              install_param_list($3, $4, LLNULL, PROCEDURE_NAME); 
	      /* if there is only a control end the control parent is not set */
              
	     }
 
	| proc_attr SUBROUTINE new_prog procname arglist
	    { install_param_list($4, $5, LLNULL, PROCEDURE_NAME); 
              if($1->variant == RECURSIVE_OP) 
                   $4->attr = $4->attr | RECURSIVE_BIT;
              pred_bfnd->entry.Template.ll_ptr3 = $1;
            }
	| FUNCTION new_prog funcname arglist opt_result_clause
	    {
              install_param_list($3, $4, $5, FUNCTION_NAME);  
  	      pred_bfnd->entry.Template.ll_ptr1 = $5;
            }
	| typedfunc arglist opt_result_clause
	    {
              install_param_list($1, $2, $3, FUNCTION_NAME); 
	      pred_bfnd->entry.Template.ll_ptr1 = $3;
	    }
	| ENTRY name arglist opt_result_clause
	    {PTR_BFND p, bif;
	     PTR_SYMB q = SMNULL;
             PTR_LLND l = LLNULL;

	     if(parstate==OUTSIDE || procclass==CLMAIN || procclass==CLBLOCK)
	        err("Misplaced ENTRY statement", 35);

	     bif = cur_scope();
	     if (bif->variant == FUNC_HEDR) {
	        q = make_function($2, bif->entry.Template.symbol->type, LOCAL);
	        l = construct_entry_list(q, $3, FUNCTION_NAME); 
             }
             else if ((bif->variant == PROC_HEDR) || 
                      (bif->variant == PROS_HEDR) || /* added for FORTRAN M */
                      (bif->variant == PROG_HEDR)) {
	             q = make_procedure($2,LOCAL);
  	             l = construct_entry_list(q, $3, PROCEDURE_NAME); 
             }
	     p = get_bfnd(fi,ENTRY_STAT, q, l, $4, LLNULL);
	     set_blobs(p, pred_bfnd, SAME_GROUP);
             q->decl = YES;   /*4.02.03*/
             q->entry.proc_decl.proc_hedr = p; /*5.02.03*/
	    }
	| MODULE  new_prog  name 
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


               s = make_module($3);
	       s->decl = YES;   /* variable declaration has been seen. */
	        if (pred_bfnd != global_bfnd)
		    err("Misplaced MODULE statement", 33);
              p = get_bfnd(fi, MODULE_STMT, s, LLNULL, LLNULL, LLNULL);
	      s->entry.Template.func_hedr = p; /* !!!????*/
	      set_blobs(p, global_bfnd, NEW_GROUP1);
	      add_scope_level(p, NO);	
	      position =  IN_MODULE;    /*IN_PROC*/
              privateall = 0;
            }
	;

new_prog:   { newprog(); 
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
	    }
	;

proc_attr: RECURSIVE needkeyword
           { $$ = make_llnd(fi, RECURSIVE_OP, LLNULL, LLNULL, SMNULL); }
         | PURE needkeyword
           { $$ = make_llnd(fi, PURE_OP, LLNULL, LLNULL, SMNULL); }
         | ELEMENTAL needkeyword
           { $$ = make_llnd(fi, ELEMENTAL_OP, LLNULL, LLNULL, SMNULL); }
         ;

procname:  name
           { PTR_BFND p;

	      $$ = make_procedure($1, LOCAL);
	      $$->decl = YES;   /* variable declaration has been seen. */
             /* if (pred_bfnd != global_bfnd)
		 {
	         err("Misplaced SUBROUTINE statement", 34);
		 }  
              */
	      p = get_bfnd(fi,PROC_HEDR, $$, LLNULL, LLNULL, LLNULL);
              $$->entry.proc_decl.proc_hedr = p;
	      set_blobs(p, pred_bfnd, NEW_GROUP1);
              add_scope_level(p, NO);
            }        
          ;

funcname:  name
           { PTR_BFND p;

	      $$ = make_function($1, TYNULL, LOCAL);
	      $$->decl = YES;   /* variable declaration has been seen. */
             /* if (pred_bfnd != global_bfnd)
	         err("Misplaced FUNCTION statement", 34); */
	      p = get_bfnd(fi,FUNC_HEDR, $$, LLNULL, LLNULL, LLNULL);
              $$->entry.func_decl.func_hedr = p;
	      set_blobs(p, pred_bfnd, NEW_GROUP1);
              add_scope_level(p, NO);
            }        
           ;

typedfunc: type  FUNCTION new_prog name 
           { PTR_BFND p;
             PTR_LLND l;

	      $$ = make_function($4, $1, LOCAL);
	      $$->decl = YES;   /* variable declaration has been seen. */
              l = make_llnd(fi, TYPE_OP, LLNULL, LLNULL, SMNULL);
              l->type = $1;
	      p = get_bfnd(fi,FUNC_HEDR, $$, LLNULL, l, LLNULL);
              $$->entry.func_decl.func_hedr = p;
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
           }        
	| type  proc_attr FUNCTION new_prog name
           { PTR_BFND p;
             PTR_LLND l;
	      $$ = make_function($5, $1, LOCAL);
	      $$->decl = YES;   /* variable declaration has been seen. */
              if($2->variant == RECURSIVE_OP)
	         $$->attr = $$->attr | RECURSIVE_BIT;
              l = make_llnd(fi, TYPE_OP, LLNULL, LLNULL, SMNULL);
              l->type = $1;
             /* if (pred_bfnd != global_bfnd)
	         err("Misplaced FUNCTION statement", 34);*/
	      p = get_bfnd(fi,FUNC_HEDR, $$, LLNULL, l, $2);
              $$->entry.func_decl.func_hedr = p;
	      set_blobs(p, pred_bfnd, NEW_GROUP1);
              add_scope_level(p, NO);
            }        
	| proc_attr FUNCTION new_prog name
           { PTR_BFND p;

	      $$ = make_function($4, TYNULL, LOCAL);
	      $$->decl = YES;   /* variable declaration has been seen. */
              if($1->variant == RECURSIVE_OP)
	        $$->attr = $$->attr | RECURSIVE_BIT;
              /*if (pred_bfnd != global_bfnd)
	         err("Misplaced FUNCTION statement",34);*/
	      p = get_bfnd(fi,FUNC_HEDR, $$, LLNULL, LLNULL, $1);
              $$->entry.func_decl.func_hedr = p;
	      set_blobs(p, pred_bfnd, NEW_GROUP1);
              add_scope_level(p, NO);
            }        
	| proc_attr type FUNCTION new_prog name
           { PTR_BFND p;
              PTR_LLND l;
	      $$ = make_function($5, $2, LOCAL);
	      $$->decl = YES;   /* variable declaration has been seen. */
              if($1->variant == RECURSIVE_OP)
	        $$->attr = $$->attr | RECURSIVE_BIT;
              l = make_llnd(fi, TYPE_OP, LLNULL, LLNULL, SMNULL);
              l->type = $2;
             /* if (pred_bfnd != global_bfnd)
	          err("Misplaced FUNCTION statement",34);*/
	      p = get_bfnd(fi,FUNC_HEDR, $$, LLNULL, l, $1);
              $$->entry.func_decl.func_hedr = p;
	      set_blobs(p, pred_bfnd, NEW_GROUP1);
              add_scope_level(p, NO);
            }        
	;

opt_result_clause: needkeyword keywordoff
	    { $$ = LLNULL; }
	| needkeyword RESULT LEFTPAR name RIGHTPAR 
	    { PTR_SYMB s;
              s = make_scalar($4, TYNULL, LOCAL);
              $$ = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
            }
	;

name:	  IDENTIFIER
	    { $$ = look_up_sym(yytext); }
	;

progname:   { $$ = make_program(look_up_sym("_MAIN")); }
	| name
	    {
              $$ = make_program($1);
	      $$->decl = YES;   /* variable declaration has been seen. */
            }
	;

blokname:   { $$ = make_program(look_up_sym("_BLOCK")); }
	| name
	    {
              $$ = make_program($1); 
	      $$->decl = YES;   /* variable declaration has been seen. */
	    }
	;

arglist:
	    { $$ = SMNULL; }
	| LEFTPAR RIGHTPAR
	    { $$ = SMNULL; }
	| LEFTPAR args RIGHTPAR
	    { $$ = $2; }
	;

args:	  arg
	| args COMMA arg
	    { $$ = set_id_list($1, $3); }
	;

arg:	  name
	    {
	      $$ = make_scalar($1, TYNULL, IO);
            }
	| ASTER
	    { $$ = make_scalar(look_up_sym("*"), TYNULL, IO); }   /*star_symb*/
	;



filename: CHAR_CONSTANT
	    { char *s;

	      s = copyn(yyleng+1, yytext);
	      s[yyleng] = '\0';
	      $$ = s;
	    }
	;

needkeyword: 
            { needkwd = 1; }
         ;

keywordoff: 
            { needkwd = NO; }
         ;

/* The scanner checks if the keyword is ONLY. */
keyword_if_colon_follow: 
            { colon_flag = YES; }
         ;

/*
 * Grammar for declarations
 */

spec:	  type_dcl
	| end_type
	| dcl
	| common
        | dimension
        | dvm_specification /* FDVM */
	| external
	| intrinsic
	| equivalence
	| implicit
	| attrib
	| namelist
	| data
	| SAVE in_dcl
	    {
	      saveall = YES;
	      $$ = get_bfnd(fi,SAVE_DECL, SMNULL, LLNULL, LLNULL, LLNULL);
	    }
	| SAVE in_dcl  opt_double_colon savelist
	    {
	      $$ = get_bfnd(fi,SAVE_DECL, SMNULL, $4, LLNULL, LLNULL);
            }

	| FORMAT inside
           { PTR_LLND p;

	      p = make_llnd(fi,STMT_STR, LLNULL, LLNULL, SMNULL);
	      p->entry.string_val = copys(stmtbuf);
	      $$ = get_bfnd(fi,FORMAT_STAT, SMNULL, p, LLNULL, LLNULL);
             }
	| PARAMETER in_dcl LEFTPAR paramlist RIGHTPAR
           { $$ = get_bfnd(fi,PARAM_DECL, SMNULL, $4, LLNULL, LLNULL); }
	| intent
	| optional
	| public
	| private
        | sequence
	| allocatable
	| pointer
	| target
	| interface
	| use_stat
	| module_proc_stmt
        | static
	;

interface: INTERFACE in_dcl
	    { $$ = get_bfnd(fi, INTERFACE_STMT, SMNULL, LLNULL, LLNULL, LLNULL); 
              add_scope_level($$, NO);     
            }
	| INTERFACE in_dcl name
	    { PTR_SYMB s;

	      s = make_procedure($3, LOCAL);
	      s->variant = INTERFACE_NAME;
	      $$ = get_bfnd(fi, INTERFACE_STMT, s, LLNULL, LLNULL, LLNULL);
              add_scope_level($$, NO);
	    }
	| INTERFACEOPERATOR in_dcl LEFTPAR operator RIGHTPAR 
	    { PTR_SYMB s;

	      s = make_function($4, global_default, LOCAL);
	      s->variant = INTERFACE_NAME;
	      $$ = get_bfnd(fi, INTERFACE_OPERATOR, s, LLNULL, LLNULL, LLNULL);
              add_scope_level($$, NO);
	    }
	| INTERFACEASSIGNMENT in_dcl LEFTPAR EQUAL RIGHTPAR
	    { PTR_SYMB s;


	      s = make_procedure(look_up_sym("="), LOCAL);
	      s->variant = INTERFACE_NAME;
	      $$ = get_bfnd(fi, INTERFACE_ASSIGNMENT, s, LLNULL, LLNULL, LLNULL);
              add_scope_level($$, NO);
	    }
	| ENDINTERFACE  opt_unit_name 
	    { parstate = INDCL;
              $$ = get_bfnd(fi, CONTROL_END, SMNULL, LLNULL, LLNULL, LLNULL); 
	      /*process_interface($$);*/ /*podd 01.02.03*/
              delete_beyond_scope_level(pred_bfnd);
	    }
	;

defined_op: DEFINED_OPERATOR
	    { $$ = look_up_sym(yytext); }
	;

operator: defined_op
	    { $$ = $1; }
	| intrinsic_op
	    { $$ = $1; }
	;

intrinsic_op: PLUS
	    { $$ = look_up_op(PLUS); }
	| MINUS
	    { $$ = look_up_op(MINUS); }
	| ASTER
	    { $$ = look_up_op(ASTER); }
	| DASTER
	    { $$ = look_up_op(DASTER); }
	| SLASH
	    { $$ = look_up_op(SLASH); }
	| DSLASH
	    { $$ = look_up_op(DSLASH); }
	| AND
	    { $$ = look_up_op(AND); }
	| OR
	    { $$ = look_up_op(OR); }
	| XOR
	    { $$ = look_up_op(XOR); }
	| NOT
	    { $$ = look_up_op(NOT); }
	| EQ
	    { $$ = look_up_op(EQ); }
	| NE
	    { $$ = look_up_op(NE); }
	| GT
	    { $$ = look_up_op(GT); }
	| GE
	    { $$ = look_up_op(GE); }
        | LT
	    { $$ = look_up_op(LT); }
	| LE
	    { $$ = look_up_op(LE); }
	| NEQV
	    { $$ = look_up_op(NEQV); }
	| EQV
	    { $$ = look_up_op(EQV); }
	;


type_dcl: TYPE_DECL in_dcl opt_double_colon name
	   {
             PTR_SYMB s;
         
             type_var = s = make_derived_type($4, TYNULL, LOCAL);	
             $$ = get_bfnd(fi, STRUCT_DECL, s, LLNULL, LLNULL, LLNULL);
             add_scope_level($$, NO);
	   }

	| TYPE_DECL COMMA in_dcl needkeyword access_spec opt_double_colon name
	   { PTR_SYMB s;
         
             type_var = s = make_derived_type($7, TYNULL, LOCAL);	
	     s->attr = s->attr | type_opt;
             $$ = get_bfnd(fi, STRUCT_DECL, s, $5, LLNULL, LLNULL);
             add_scope_level($$, NO);
	   }
	;

end_type: ENDTYPE in_dcl
	   {
	     $$ = get_bfnd(fi, CONTROL_END, SMNULL, LLNULL, LLNULL, LLNULL);
	     if (type_var != SMNULL)
               process_type(type_var, $$);
             type_var = SMNULL;
	     delete_beyond_scope_level(pred_bfnd);
           }
	| ENDTYPE in_dcl name
	   {
             $$ = get_bfnd(fi, CONTROL_END, SMNULL, LLNULL, LLNULL, LLNULL);
	     if (type_var != SMNULL)
               process_type(type_var, $$);
             type_var = SMNULL;
	     delete_beyond_scope_level(pred_bfnd);	
           }
	;

dcl:	 type options name in_dcl dims lengspec initial_value
	    { 
	      PTR_LLND q, r, l;
	     /* PTR_SYMB s;*/
	      PTR_TYPE t;
	      int type_opts;

	      vartype = $1;
              if($6 && vartype->variant != T_STRING)
                errstr("Non character entity  %s  has length specification",$3->ident,41);
              t = make_type_node(vartype, $6);
	      type_opts = type_options;
	      if ($5) type_opts = type_opts | DIMENSION_BIT;
	      if ($5)
		 q = deal_with_options($3, t, type_opts, $5, ndim, $7, $5);
	      else q = deal_with_options($3, t, type_opts, attr_dims, attr_ndim, $7, $5);
	      r = make_llnd(fi, EXPR_LIST, q, LLNULL, SMNULL);
	      l = make_llnd(fi, TYPE_OP, LLNULL, LLNULL, SMNULL);
	      l->type = vartype;
	      $$ = get_bfnd(fi,VAR_DECL, SMNULL, r, l, $2);
	    }
	| dcl COMMA name dims lengspec initial_value
	    { 
	      PTR_LLND q, r;
	    /*  PTR_SYMB s;*/
              PTR_TYPE t;
	      int type_opts;
              if($5 && vartype->variant != T_STRING)
                errstr("Non character entity  %s  has length specification",$3->ident,41);
              t = make_type_node(vartype, $5);
	      type_opts = type_options;
	      if ($4) type_opts = type_opts | DIMENSION_BIT;
	      if ($4)
		 q = deal_with_options($3, t, type_opts, $4, ndim, $6, $4);
	      else q = deal_with_options($3, t, type_opts, attr_dims, attr_ndim, $6, $4);
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      add_to_lowLevelList(r, $1->entry.Template.ll_ptr1);
       	    }
        ;

options: 
	    { $$ = LLNULL; }
	| COLON COLON
            { $$ = LLNULL; }
	| COMMA needkeyword attr_spec_list COLON COLON
	    { $$ = $3; }
	;

attr_spec_list:  attr_spec
	    { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	| attr_spec_list COMMA needkeyword attr_spec
	    { $$ = set_ll_list($1, $4, EXPR_LIST); }
	;

attr_spec: PARAMETER
	    { type_options = type_options | PARAMETER_BIT; 
              $$ = make_llnd(fi, PARAMETER_OP, LLNULL, LLNULL, SMNULL);
            }
	| access_spec
            { $$ = $1; }
	| ALLOCATABLE
	    { type_options = type_options | ALLOCATABLE_BIT;
              $$ = make_llnd(fi, ALLOCATABLE_OP, LLNULL, LLNULL, SMNULL);
	    }
	| DIMENSION dims
	    { type_options = type_options | DIMENSION_BIT;
	      attr_ndim = ndim;
	      attr_dims = $2;
              $$ = make_llnd(fi, DIMENSION_OP, $2, LLNULL, SMNULL);
            }
	| EXTERNAL
	    { type_options = type_options | EXTERNAL_BIT;
              $$ = make_llnd(fi, EXTERNAL_OP, LLNULL, LLNULL, SMNULL);
            }
	| INTENT LEFTPAR intent_spec RIGHTPAR
	    { $$ = $3; }
	| INTRINSIC
	    { type_options = type_options | INTRINSIC_BIT;
              $$ = make_llnd(fi, INTRINSIC_OP, LLNULL, LLNULL, SMNULL);
            }
	| OPTIONAL
	    { type_options = type_options | OPTIONAL_BIT;
              $$ = make_llnd(fi, OPTIONAL_OP, LLNULL, LLNULL, SMNULL);
            }
	| POINTER
	    { type_options = type_options | POINTER_BIT;
              $$ = make_llnd(fi, POINTER_OP, LLNULL, LLNULL, SMNULL);
            }
	| SAVE
	    { type_options = type_options | SAVE_BIT; 
              $$ = make_llnd(fi, SAVE_OP, LLNULL, LLNULL, SMNULL);
            }
	| STATIC
	    { type_options = type_options | SAVE_BIT; 
              $$ = make_llnd(fi, STATIC_OP, LLNULL, LLNULL, SMNULL);
            }
	| TARGET
	    { type_options = type_options | TARGET_BIT; 
              $$ = make_llnd(fi, TARGET_OP, LLNULL, LLNULL, SMNULL);
            }
	;

intent_spec: needkeyword IN
	    { type_options = type_options | IN_BIT;  type_opt = IN_BIT; 
              $$ = make_llnd(fi, IN_OP, LLNULL, LLNULL, SMNULL);
            }
        | needkeyword OUT
	    { type_options = type_options | OUT_BIT;  type_opt = OUT_BIT; 
              $$ = make_llnd(fi, OUT_OP, LLNULL, LLNULL, SMNULL);
            }
        | needkeyword INOUT
	    { type_options = type_options | INOUT_BIT;  type_opt = INOUT_BIT;
              $$ = make_llnd(fi, INOUT_OP, LLNULL, LLNULL, SMNULL);
            }
	;

access_spec: PUBLIC
	    { type_options = type_options | PUBLIC_BIT; 
              type_opt = PUBLIC_BIT;
              $$ = make_llnd(fi, PUBLIC_OP, LLNULL, LLNULL, SMNULL);
            }
	   | PRIVATE
	    { type_options =  type_options | PRIVATE_BIT;
               type_opt = PRIVATE_BIT;
              $$ = make_llnd(fi, PRIVATE_OP, LLNULL, LLNULL, SMNULL);
            }
        ;

intent:	 INTENT in_dcl LEFTPAR intent_spec RIGHTPAR opt_double_colon name
	    { 
	      PTR_LLND q, r;
	      PTR_SYMB s;

              s = make_scalar($7, TYNULL, LOCAL);
	      s->attr = s->attr | type_opt;	
	      q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      $$ = get_bfnd(fi, INTENT_STMT, SMNULL, r, $4, LLNULL);
	    }
	| intent COMMA name 
	    { 
	      PTR_LLND q, r;
	      PTR_SYMB s;

              s = make_scalar($3, TYNULL, LOCAL);	
	      s->attr = s->attr | type_opt;
	      q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      add_to_lowLevelList(r, $1->entry.Template.ll_ptr1);
  	    }
	;

optional: OPTIONAL in_dcl opt_double_colon name
	    { 
	      PTR_LLND q, r;
	      PTR_SYMB s;

              s = make_scalar($4, TYNULL, LOCAL);	
	      s->attr = s->attr | OPTIONAL_BIT;
	      q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      $$ = get_bfnd(fi, OPTIONAL_STMT, SMNULL, r, LLNULL, LLNULL);
	    }
	| optional COMMA name 
	    { 
	      PTR_LLND q, r;
	      PTR_SYMB s;

              s = make_scalar($3, TYNULL, LOCAL);	
	      s->attr = s->attr | OPTIONAL_BIT;
	      q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      add_to_lowLevelList(r, $1->entry.Template.ll_ptr1);
  	    }
	;

static: STATIC in_dcl opt_double_colon var
	    { 
	      PTR_LLND r;
	      PTR_SYMB s;

              s = $4->entry.Template.symbol; 
              s->attr = s->attr | SAVE_BIT;
	      r = make_llnd(fi,EXPR_LIST, $4, LLNULL, SMNULL);
	      $$ = get_bfnd(fi, STATIC_STMT, SMNULL, r, LLNULL, LLNULL);
	    }
	| static COMMA var 
	    { 
	      PTR_LLND r;
	      PTR_SYMB s;

              s = $3->entry.Template.symbol;
              s->attr = s->attr | SAVE_BIT;
	      r = make_llnd(fi,EXPR_LIST, $3, LLNULL, SMNULL);
	      add_to_lowLevelList(r, $1->entry.Template.ll_ptr1);
  	    }
	;


private: PRIVATE in_dcl
	    {
	      privateall = 1;
	      $$ = get_bfnd(fi, PRIVATE_STMT, SMNULL, LLNULL, LLNULL, LLNULL);
	    }
	| PRIVATE in_dcl opt_double_colon private_attr use_name_list
	    {
	      /*type_options = type_options | PRIVATE_BIT;*/
	      $$ = get_bfnd(fi, PRIVATE_STMT, SMNULL, $5, LLNULL, LLNULL);
            }
        ;
private_attr:
              {type_opt = PRIVATE_BIT;}
            ;

sequence: SEQUENCE in_dcl
	    { 
	      $$ = get_bfnd(fi, SEQUENCE_STMT, SMNULL, LLNULL, LLNULL, LLNULL);
            }

public: PUBLIC in_dcl
	    {
	      /*saveall = YES;*/ /*14.03.03*/
	      $$ = get_bfnd(fi, PUBLIC_STMT, SMNULL, LLNULL, LLNULL, LLNULL);
	    }
	| PUBLIC in_dcl opt_double_colon public_attr use_name_list
	    {
	      /*type_options = type_options | PUBLIC_BIT;*/
	      $$ = get_bfnd(fi, PUBLIC_STMT, SMNULL, $5, LLNULL, LLNULL);
            }

public_attr:
              {type_opt = PUBLIC_BIT;}
            ;

type:	  typespec  opt_key_hedr  selector  opt_key_hedr 
	    {
	      type_options = 0;
              /* following block added by dbg */
	      ndim = 0;
	      attr_ndim = 0;
	      attr_dims = LLNULL;
	      /* end section added by dbg */
              $$ = make_type_node($1, $3);
            }
	| TYPE LEFTPAR name RIGHTPAR opt_key_hedr
	    { PTR_TYPE t;

	      type_options = 0;
	      ndim = 0;
	      attr_ndim = 0;
	      attr_dims = LLNULL;
              t = lookup_type($3);
	      vartype = t;
	      $$ = make_type_node(t, LLNULL);
            }
	;

opt_key_hedr:
            {opt_kwd_hedr = YES;}
            ;

attrib:   att_type name

	    { PTR_TYPE p;
	      PTR_LLND q;
	      PTR_SYMB s;
              s = $2->id_attr;
	      if (s)
		   s->attr = $1;
	      else {
		p = undeftype ? global_unknown : impltype[*$2->ident - 'a'];
                s = install_entry($2, SOFT);
		s->attr = $1;
                set_type(s, p, LOCAL);
	      }
	      q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, $2->id_attr);
	      q = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      $$ = get_bfnd(fi,ATTR_DECL, SMNULL, q, LLNULL, LLNULL);
	    }

	| attrib COMMA name

	    { PTR_TYPE p;
	      PTR_LLND q, r;
	      PTR_SYMB s;
	      int att;

	      att = $1->entry.Template.ll_ptr1->entry.Template.ll_ptr1->
		    entry.Template.symbol->attr;
              s = $3->id_attr;
	      if (s)
		   s->attr = att;
	      else {
		p = undeftype ? global_unknown : impltype[*$3->ident - 'a'];
                s = install_entry($3, SOFT);
		s->attr = att;
                set_type(s, p, LOCAL);
	      }
	      q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, $3->id_attr);
	      for (r = $1->entry.Template.ll_ptr1;
		   r->entry.list.next;
		   r = r->entry.list.next) ;
	      r->entry.list.next = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);

	    }
	;

att_type: GLOBAL_A
	 { $$ = ATT_GLOBAL; }
	| CLUSTER
	 { $$ = ATT_CLUSTER; }
	;

/*
opt_attr:
	| TASK_GLOBAL
	| PROCESS_GLOBAL
	| TASK_CLUSTER
	| PROCESS_CLUSTER
	;
*/
typespec:  typename
		{
/*		  varleng = ($1<0 || $1==TYLONG ? 0 : typesize[$1]); */
		  vartype = $1;
		}

	;

typename: INTEGER	{ $$ = global_int; }
	| REAL		{ $$ = global_float; }
	| COMPLEX	{ $$ = global_complex; }
	| DOUBLEPRECISION { $$ = global_double; }
	| DOUBLECOMPLEX	{ $$ = global_dcomplex; }
	| LOGICAL	{ $$ = global_bool; }
	| CHARACTER	{ $$ = global_string; }

	;

lengspec:
	    { $$ = LLNULL; }
	| proper_lengspec
	    { $$ = $1; }
	;

proper_lengspec: ASTER intonlyon integer_constant intonlyoff  opt_key_hedr
	        { $$ = make_llnd(fi, LEN_OP, $3, LLNULL, SMNULL); }
	       | ASTER  intonlyon LEFTPAR intonlyoff ASTER RIGHTPAR 
	        { PTR_LLND l;

                 l = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL); 
                 l->entry.string_val = (char *)"*";
                 $$ = make_llnd(fi, LEN_OP, l,l, SMNULL);
                }
               | ASTER intonlyon LEFTPAR  intonlyoff expr RIGHTPAR 
                {$$ = make_llnd(fi, LEN_OP, $5, $5, SMNULL);}
	       ;

selector:
	    { $$ = LLNULL; }
 	| proper_lengspec  
	    { $$ = $1; } 
        | LEFTPAR in_ioctl  clause  end_ioctl RIGHTPAR 
	    { /*$$ = make_llnd(fi, PAREN_OP, $2, LLNULL, SMNULL);*/  $$ = $3;  }
        | LEFTPAR  in_ioctl  clause  end_ioctl COMMA  in_ioctl  clause  end_ioctl RIGHTPAR
      /*  | LEFTPAR in_ioctl clause COMMA clause RIGHTPAR 
	    { PTR_LLND l;

              l = make_llnd(fi, CONS, $2, $4, SMNULL);
	      $$ = make_llnd(fi, PAREN_OP, l, LLNULL, SMNULL);}
       */
            { if($7->variant==LENGTH_OP && $3->variant==$7->variant)
                $7->variant=KIND_OP;
                $$ = make_llnd(fi, CONS, $3, $7, SMNULL); 
            }             
	;

clause:   expr
	    { if(vartype->variant == T_STRING)
                $$ = make_llnd(fi,LENGTH_OP,$1,LLNULL,SMNULL);
              else
                $$ = make_llnd(fi,KIND_OP,$1,LLNULL,SMNULL);
            }
	| ASTER
	    { PTR_LLND l;
	      l = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
	      l->entry.string_val = (char *)"*";
              $$ = make_llnd(fi,LENGTH_OP,l,LLNULL,SMNULL);
            }
	| nameeq expr
	    { /* $$ = make_llnd(fi, SPEC_PAIR, $2, LLNULL, SMNULL); */
	     char *q;
             q = $1->entry.string_val;
  	     if (strcmp(q, "len") == 0)
               $$ = make_llnd(fi,LENGTH_OP,$2,LLNULL,SMNULL);
             else
                $$ = make_llnd(fi,KIND_OP,$2,LLNULL,SMNULL);              
            }
        | nameeq ASTER
	    { PTR_LLND l;
	      l = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
	      l->entry.string_val = (char *)"*";
              $$ = make_llnd(fi,LENGTH_OP,l,LLNULL,SMNULL);
            }
	;

end_ioctl:
          {endioctl();}
         ;

/*
int_nameeq_on:
	    { intonly = inioctl = YES; }
	;

int_nameeq_off:
	    { intonly = inioctl = NO; }
	;
*/
initial_value: 
	    { $$ = LLNULL; }
	| EQUAL expr
	    { $$ = $2; }

        | POINT_TO expr
            { $$ = make_llnd(fi, POINTST_OP, LLNULL, $2, SMNULL); }
	;

dimension: DIMENSION  opt_double_colon  in_dcl   name dims 
	    { PTR_SYMB s;
	      PTR_LLND q, r;
	      if(! $5) {
		err("No dimensions in DIMENSION statement", 42);
	      }
              if(statement_kind == 1) /*DVM-directive*/
                err("No shape specification", 65);                
	      s = make_array($4, TYNULL, $5, ndim, LOCAL);
	      s->attr = s->attr | DIMENSION_BIT;
	      q = make_llnd(fi,ARRAY_REF, $5, LLNULL, s);
	      s->type->entry.ar_decl.ranges = $5;
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      $$ = get_bfnd(fi,DIM_STAT, SMNULL, r, LLNULL, LLNULL);
	    }
	| dimension COMMA name dims 
           {  PTR_SYMB s;
	      PTR_LLND q, r;
	      if(! $4) {
		err("No dimensions in DIMENSION statement", 42);
	      }
	      s = make_array($3, TYNULL, $4, ndim, LOCAL);
	      s->attr = s->attr | DIMENSION_BIT;
	      q = make_llnd(fi,ARRAY_REF, $4, LLNULL, s);
	      s->type->entry.ar_decl.ranges = $4;
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      add_to_lowLevelList(r, $1->entry.Template.ll_ptr1);
	}
	;

allocatable: ALLOCATABLE in_dcl opt_double_colon var
                   /* ALLOCATABLE in_dcl opt_double_colon name dims lengspec*/
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
              $4->entry.Template.symbol->attr = $4->entry.Template.symbol->attr | ALLOCATABLE_BIT;
	      r = make_llnd(fi,EXPR_LIST, $4, LLNULL, SMNULL);
	      $$ = get_bfnd(fi, ALLOCATABLE_STMT, SMNULL, r, LLNULL, LLNULL);
	    }
	| allocatable COMMA var
                /*allocatable COMMA name dims lengspec */
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
              $3->entry.Template.symbol->attr = $3->entry.Template.symbol->attr | ALLOCATABLE_BIT;
              r = make_llnd(fi,EXPR_LIST, $3, LLNULL, SMNULL);
	      add_to_lowLevelList(r, $1->entry.Template.ll_ptr1);
	}
	;

pointer: POINTER in_dcl opt_double_colon var
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
              s = $4->entry.Template.symbol; /*17.02.03*/
              s->attr = s->attr | POINTER_BIT;
	      r = make_llnd(fi,EXPR_LIST, $4, LLNULL, SMNULL);
	      $$ = get_bfnd(fi, POINTER_STMT, SMNULL, r, LLNULL, LLNULL);
	    }
	| pointer COMMA var
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
              s = $3->entry.Template.symbol; /*17.02.03*/
              s->attr = s->attr | POINTER_BIT;
	      r = make_llnd(fi,EXPR_LIST, $3, LLNULL, SMNULL);
	      add_to_lowLevelList(r, $1->entry.Template.ll_ptr1);
	}
	;

target: TARGET in_dcl opt_double_colon var
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
              $4->entry.Template.symbol->attr = $4->entry.Template.symbol->attr | TARGET_BIT;
	      r = make_llnd(fi,EXPR_LIST, $4, LLNULL, SMNULL);
	      $$ = get_bfnd(fi, TARGET_STMT, SMNULL, r, LLNULL, LLNULL);
	    }
	| target COMMA var
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
              $3->entry.Template.symbol->attr = $3->entry.Template.symbol->attr | TARGET_BIT;
	      r = make_llnd(fi,EXPR_LIST, $3, LLNULL, SMNULL);
	      add_to_lowLevelList(r, $1->entry.Template.ll_ptr1);
	}
	;

common:	  COMMON in_dcl var
	    { PTR_LLND p, q;

              p = make_llnd(fi,EXPR_LIST, $3, LLNULL, SMNULL);
	      q = make_llnd(fi,COMM_LIST, p, LLNULL, SMNULL);
	      $$ = get_bfnd(fi,COMM_STAT, SMNULL, q, LLNULL, LLNULL);
	    }
	| COMMON in_dcl comblock var
	    { PTR_LLND p, q;

              p = make_llnd(fi,EXPR_LIST, $4, LLNULL, SMNULL);
	      q = make_llnd(fi,COMM_LIST, p, LLNULL, $3);
	      $$ = get_bfnd(fi,COMM_STAT, SMNULL, q, LLNULL, LLNULL);
	    }
	| common opt_comma comblock opt_comma var
	    { PTR_LLND p, q;

              p = make_llnd(fi,EXPR_LIST, $5, LLNULL, SMNULL);
	      q = make_llnd(fi,COMM_LIST, p, LLNULL, $3);
	      add_to_lowList(q, $1->entry.Template.ll_ptr1);
	    }
	| common COMMA var
	    { PTR_LLND p, r;

              p = make_llnd(fi,EXPR_LIST, $3, LLNULL, SMNULL);
	      /*q = make_llnd(fi,COMM_LIST, p, LLNULL, SMNULL);*/
	      for (r = $1->entry.Template.ll_ptr1;
		   r->entry.list.next;
		   r = r->entry.list.next);
	      add_to_lowLevelList(p, r->entry.Template.ll_ptr1);
	    }
	;


namelist:  NAMELIST in_dcl namelist_group ident
	    { PTR_LLND q, r;

              q = make_llnd(fi,EXPR_LIST, $4, LLNULL, SMNULL);
	      r = make_llnd(fi,NAMELIST_LIST, q, LLNULL, $3);
	      $$ = get_bfnd(fi,NAMELIST_STAT, SMNULL, r, LLNULL, LLNULL);
	    }
	| namelist opt_comma namelist_group opt_comma ident
	    { PTR_LLND q, r;

              q = make_llnd(fi,EXPR_LIST, $5, LLNULL, SMNULL);
	      r = make_llnd(fi,NAMELIST_LIST, q, LLNULL, $3);
	      add_to_lowList(r, $1->entry.Template.ll_ptr1);
	    }
	| namelist COMMA ident
	    { PTR_LLND q, r;

              q = make_llnd(fi,EXPR_LIST, $3, LLNULL, SMNULL);
	      for (r = $1->entry.Template.ll_ptr1;
		   r->entry.list.next;
		   r = r->entry.list.next);
	      add_to_lowLevelList(q, r->entry.Template.ll_ptr1);
	    }
	;

namelist_group: SLASH name SLASH
	    { $$ =  make_local_entity($2, NAMELIST_NAME,global_default,LOCAL); }
	;

comblock: DSLASH
	    { $$ = NULL; /*make_common(look_up_sym("*"));*/ }
	| SLASH name SLASH
	    { $$ = make_common($2); }
	;


var:	  name dims
          {  PTR_SYMB s;
	
	      if($2) {
		s = make_array($1, TYNULL, $2, ndim, LOCAL);
                s->attr = s->attr | DIMENSION_BIT;
		s->type->entry.ar_decl.ranges = $2;
		$$ = make_llnd(fi,ARRAY_REF, $2, LLNULL, s);
	      }
	      else {
		s = make_scalar($1, TYNULL, LOCAL);	
		$$ = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
	      }

          }
            
	;

external: EXTERNAL in_dcl  opt_double_colon  name
	    { PTR_LLND p, q;
              PTR_SYMB s;

	      s = make_external($4, TYNULL);
	      s->attr = s->attr | EXTERNAL_BIT;
              q = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
	      p = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      $$ = get_bfnd(fi,EXTERN_STAT, SMNULL, p, LLNULL, LLNULL);
	    }

	| external COMMA name
	    { PTR_LLND p, q;
              PTR_SYMB s;

	      s = make_external($3, TYNULL);
	      s->attr = s->attr | EXTERNAL_BIT;
              p = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
	      q = make_llnd(fi,EXPR_LIST, p, LLNULL, SMNULL);
	      add_to_lowLevelList(q, $1->entry.Template.ll_ptr1);
	    }
	;

intrinsic: INTRINSIC in_dcl  opt_double_colon  name
	    { PTR_LLND p, q;
              PTR_SYMB s;

	      s = make_intrinsic($4, TYNULL); /*make_function($3, TYNULL, NO);*/
	      s->attr = s->attr | INTRINSIC_BIT;
              q = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
	      p = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      $$ = get_bfnd(fi,INTRIN_STAT, SMNULL, p,
			     LLNULL, LLNULL);
	    }

	| intrinsic COMMA name
	    { PTR_LLND p, q;
              PTR_SYMB s;

	      s = make_intrinsic($3, TYNULL); /* make_function($3, TYNULL, NO);*/
	      s->attr = s->attr | INTRINSIC_BIT;
              p = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
	      q = make_llnd(fi,EXPR_LIST, p, LLNULL, SMNULL);
	      add_to_lowLevelList(q, $1->entry.Template.ll_ptr1);
	    }

	;


equivalence:  EQUIVALENCE in_dcl equivset
	    {
	      $$ = get_bfnd(fi,EQUI_STAT, SMNULL, $3,
			     LLNULL, LLNULL);
	    }

	| equivalence COMMA equivset
	    { 
	      add_to_lowLevelList($3, $1->entry.Template.ll_ptr1);
	    }

	;

equivset: LEFTPAR equivlist RIGHTPAR
	  {
	      $$ = make_llnd(fi,EQUI_LIST, $2, LLNULL, SMNULL);
           }
	;

equivlist: equi_object COMMA equi_object
           { PTR_LLND p;
	      p = make_llnd(fi,EXPR_LIST, $3, LLNULL, SMNULL);
	      $$ = make_llnd(fi,EXPR_LIST, $1, p, SMNULL);
	    }

	| equivlist COMMA equi_object
	    { PTR_LLND p;

	      p = make_llnd(fi,EXPR_LIST, $3, LLNULL, SMNULL);
	      add_to_lowLevelList(p, $1);
	    }
	;

equi_object: name
        {  PTR_SYMB s;
           s=make_scalar($1,TYNULL,LOCAL);
           $$ = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
           s->attr = s->attr | EQUIVALENCE_BIT;
            /*$$=$1; $$->entry.Template.symbol->attr = $$->entry.Template.symbol->attr | EQUIVALENCE_BIT; */
        }
           | name LEFTPAR subscript_list RIGHTPAR
        {  PTR_SYMB s;
           s=make_array($1,TYNULL,LLNULL,0,LOCAL);
           $$ = make_llnd(fi,ARRAY_REF, $3, LLNULL, s);
           s->attr = s->attr | EQUIVALENCE_BIT;
            /*$$->entry.Template.symbol->attr = $$->entry.Template.symbol->attr | EQUIVALENCE_BIT; */
        }

     /*   { $$ = $1;
         $$->entry.Template.symbol->attr = $$->entry.Template.symbol->attr | EQUIVALENCE_BIT;
	 $$->variant == ARRAY_REF; 
         $$->entry.Template.ll_ptr1 = $3;
        }
      */
        ;

           | asubstring
           ;

data:	  data1
            { PTR_LLND p;
              data_stat = NO;
	      p = make_llnd(fi,STMT_STR, LLNULL, LLNULL,
			    SMNULL);
              p->entry.string_val = copys(stmtbuf);
	      $$ = get_bfnd(fi,DATA_DECL, SMNULL, p, LLNULL, LLNULL);
            }

data1:	  DATA inside data_in datapair

	| data1 opt_comma datapair
	;

data_in:
        {data_stat = YES;}
       ;

in_data:  
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
	    }
	;

datapair: datalvals SLASH datarvals SLASH

	;

datalvals: datalval

	| datalvals COMMA datalval

	;

datarvals: datarval
	| datarvals COMMA datarval
	;

datalval: data_null dataname

	| data_null dataname datasubs

	| data_null dataname datarange

	| data_null dataname datasubs datarange

	| data_null dataimplieddo
	;
data_null: 
           {;}
         ;

d_name: name
	    { $$= make_scalar($1, TYNULL, LOCAL);}
      ;

dataname: name
	    { $$= make_scalar($1, TYNULL, LOCAL); 
              $$->attr = $$->attr | DATA_BIT; 
            }
	;

datasubs: LEFTPAR iconexprlist RIGHTPAR
            { $$ = make_llnd(fi, DATA_SUBS, $2, LLNULL, SMNULL); }
	;

datarange: LEFTPAR opticonexpr COLON opticonexpr RIGHTPAR
            { $$ = make_llnd(fi, DATA_RANGE, $2, $4, SMNULL); }
	;

iconexprlist: iconexpr
	    { $$ = $1; }
	| iconexprlist COMMA iconexpr
            { $$ = add_to_lowLevelList($3, $1); }
	;

opticonexpr:
            { $$ = LLNULL; }
	| iconexpr
	    { $$ = $1; }
	;

dataimplieddo:	LEFTPAR dlist COMMA d_name EQUAL iconexprlist RIGHTPAR
            {$$= make_llnd(fi, DATA_IMPL_DO, $2, $6, $4); }
	;

dlist:	  dataelt
	    { $$ = $1; }
	| dlist COMMA dataelt
            { $$ = add_to_lowLevelList($3, $1); }
	;

dataelt:  dataname datasubs
            { $$ = make_llnd(fi, DATA_ELT, $2, LLNULL, $1); }
	| dataname datarange
            { $$ = make_llnd(fi, DATA_ELT, $2, LLNULL, $1); }
	| dataname datasubs datarange
	    {
              $2->entry.Template.ll_ptr2 = $3;
              $$ = make_llnd(fi, DATA_ELT, $2, LLNULL, $1); 
            }
	| dataimplieddo
	    { $$ = make_llnd(fi, DATA_ELT, $1, LLNULL, SMNULL); }
	;

datarval: datavalue

	| data_null d_name ASTER datavalue

	| unsignedint ASTER datavalue
	;

datavalue: data_null d_name

	| int_const
	| real_const
	| complex_const_data
	| TTRUE 
	| TTRUE  UNDER kind
	| FFALSE
	| FFALSE UNDER kind 
	| CHAR_CONSTANT
/*	| STRING	*/
/*	| bit_const	*/
        | BOZ_const
        | data_null ident LEFTPAR in_ioctl  funarglist RIGHTPAR
           {if($2->entry.Template.symbol->variant != TYPE_NAME)
               errstr("Undefined type %s",$2->entry.Template.symbol->ident,319); 
           }
	;

BOZ_const: BOZ_CONSTANT
         ;

int_const: unsignedint
	| PLUS unsignedint

	| MINUS unsignedint
				
	;

unsignedint: INT_CONSTANT
           | INT_CONSTANT UNDER kind
	   ;

real_const: unsignedreal
	  | PLUS unsignedreal

	  | MINUS unsignedreal

	  ;

unsignedreal: REAL_CONSTANT
            | REAL_CONSTANT UNDER kind 
	    | DP_CONSTANT
	    | DP_CONSTANT   UNDER kind 
	    ;
complex_const_data:	LEFTPAR complex_part COMMA complex_part RIGHTPAR
                  ;

complex_part: real_const
            | int_const
            ;

/*
bit_const:HEX_CONSTANT	{ $$ = mkbitcon(4, yyleng, yytext); }
	| OCT_CONSTANT	{ $$ = mkbitcon(3, yyleng, yytext); }
	| BITCON	{ $$ = mkbitcon(1, yyleng, yytext); }
	;
*/
iconexpr: iconterm
	    { $$ = make_llnd(fi,ICON_EXPR, $1, LLNULL, SMNULL); }
	| PLUS iconterm
            {
              PTR_LLND p;

              p = intrinsic_op_node("+", UNARY_ADD_OP, $2, LLNULL);
              $$ = make_llnd(fi,ICON_EXPR, p, LLNULL, SMNULL);
            }
	| MINUS iconterm
            {
              PTR_LLND p;
 
              p = intrinsic_op_node("-", MINUS_OP, $2, LLNULL);
              $$ = make_llnd(fi,ICON_EXPR, p, LLNULL, SMNULL);
            }
	| iconexpr PLUS iconterm
            {
              PTR_LLND p;
 
              p = intrinsic_op_node("+", ADD_OP, $1, $3);
              $$ = make_llnd(fi,ICON_EXPR, p, LLNULL, SMNULL);
            }
	| iconexpr MINUS iconterm
            {
              PTR_LLND p;
 
              p = intrinsic_op_node("-", SUBT_OP, $1, $3);
              $$ = make_llnd(fi,ICON_EXPR, p, LLNULL, SMNULL);
            }
	;

iconterm: iconfactor
            { $$ = $1; }
	| iconterm ASTER iconfactor
            { $$ = intrinsic_op_node("*", MULT_OP, $1, $3); }
	| iconterm SLASH iconfactor
            { $$ = intrinsic_op_node("/", DIV_OP, $1, $3); }
	;

iconfactor: iconprimary
	    { $$ = $1; }
	| iconprimary DASTER iconfactor
            { $$ = intrinsic_op_node("**", EXP_OP, $1, $3); }
	;

iconprimary: INT_CONSTANT
            {
              PTR_LLND p;

              p = make_llnd(fi,INT_VAL, LLNULL, LLNULL, SMNULL);
              p->entry.ival = atoi(yytext);
              p->type = global_int;
              $$ = make_llnd(fi,EXPR_LIST, p, LLNULL, SMNULL);
            }
	| d_name
            {
              PTR_LLND p;
 
              p = make_llnd(fi,VAR_REF, LLNULL, LLNULL, $1);
              $$ = make_llnd(fi,EXPR_LIST, p, LLNULL, SMNULL);
            }
	| LEFTPAR iconexpr RIGHTPAR
            {
              $$ = make_llnd(fi,EXPR_LIST, $2, LLNULL, SMNULL);
            }
	;


savelist: saveitem
         { $$ = make_llnd(fi,EXPR_LIST, $1, LLNULL, SMNULL); }
	| savelist COMMA saveitem
         { $$ = set_ll_list($1, $3, EXPR_LIST); }
	;

saveitem: var
	   { $$ = $1;
             $$->entry.Template.symbol->attr = $$->entry.Template.symbol->attr | SAVE_BIT;
           }
	| comblock
          { $$ = make_llnd(fi,COMM_LIST, LLNULL, LLNULL, $1); 
            $$->entry.Template.symbol->attr = $$->entry.Template.symbol->attr | SAVE_BIT;
          }
	;

use_name_list: use_key_word  use_name  no_use_key_word
            { $$ = set_ll_list($2, LLNULL, EXPR_LIST); }
	| use_name_list COMMA use_key_word  use_name  no_use_key_word 
	    { $$ = set_ll_list($1, $4, EXPR_LIST); }
	;

use_key_word:
            { as_op_kwd_ = YES; }
            ;

no_use_key_word:
            { as_op_kwd_ = NO; }
	     ;	     


use_name: name
	   { 
             PTR_SYMB s; 
             s = make_scalar($1, TYNULL, LOCAL);	
	     s->attr = s->attr | type_opt;
	     $$ = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
            }
	|  OPERATOR LEFTPAR operator RIGHTPAR 
	    { PTR_SYMB s;
	      s = make_function($3, global_default, LOCAL);
	      s->variant = INTERFACE_NAME;
              s->attr = s->attr | type_opt;
              $$ = make_llnd(fi,OPERATOR_OP, LLNULL, LLNULL, s);
	    }
        | ASSIGNMENT  LEFTPAR EQUAL RIGHTPAR
	    { PTR_SYMB s;
	      s = make_procedure(look_up_sym("="), LOCAL);
	      s->variant = INTERFACE_NAME;
              s->attr = s->attr | type_opt;
              $$ = make_llnd(fi,ASSIGNMENT_OP, LLNULL, LLNULL, s);
	    }
         ;


paramlist:paramitem
            { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	| paramlist COMMA paramitem
	    { $$ = set_ll_list($1, $3, EXPR_LIST); }
	;

paramitem: name EQUAL expr
	    { PTR_SYMB p;

                /* The check if name and expr have compatible types has
                   not been done yet. */ 
		p = make_constant($1, TYNULL);
 	        p->attr = p->attr | PARAMETER_BIT;
                p->entry.const_value = $3;
		$$ = make_llnd(fi,CONST_REF, LLNULL, LLNULL, p);
	    }
	;

module_proc_stmt: MODULE_PROCEDURE  proc_name_list
	    { $$ = get_bfnd(fi, MODULE_PROC_STMT, SMNULL, $2, LLNULL, LLNULL); }

proc_name_list: name
	    { PTR_SYMB s;
 	      PTR_LLND q;

	      s = make_function($1, TYNULL, LOCAL);
	      s->variant = ROUTINE_NAME;
              q = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
	      $$ = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	    }
	| proc_name_list COMMA name
	    { PTR_LLND p, q;
              PTR_SYMB s;

	      s = make_function($3, TYNULL, LOCAL);
	      s->variant = ROUTINE_NAME;
              p = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
	      q = make_llnd(fi,EXPR_LIST, p, LLNULL, SMNULL);
	      add_to_lowLevelList(q, $1);
	    }
	;


use_stat: USE in_dcl module_name 
	    { $$ = get_bfnd(fi, USE_STMT, $3, LLNULL, LLNULL, LLNULL);
              /*add_scope_level($3->entry.Template.func_hedr, YES);*/ /*17.06.01*/
              copy_module_scope($3,LLNULL); /*17.03.03*/
              colon_flag = NO;
            }
	| USE in_dcl module_name COMMA keyword_if_colon_follow rename_list
	    { $$ = get_bfnd(fi, USE_STMT, $3, $6, LLNULL, LLNULL); 
              /*add_scope_level(module_scope, YES); *//* 17.06.01*/
              copy_module_scope($3,$6); /*17.03.03 */
              colon_flag = NO;
            }
	| USE in_dcl module_name COMMA keyword_if_colon_follow ONLY 
	    { PTR_LLND l;

	      l = make_llnd(fi, ONLY_NODE, LLNULL, LLNULL, SMNULL);
              $$ = get_bfnd(fi, USE_STMT, $3, l, LLNULL, LLNULL);
            }
	| USE in_dcl module_name COMMA keyword_if_colon_follow ONLY only_list
	    { PTR_LLND l;

	      l = make_llnd(fi, ONLY_NODE, $7, LLNULL, SMNULL);
              $$ = get_bfnd(fi, USE_STMT, $3, l, LLNULL, LLNULL);
            }
	;

module_name: name
            {
              if ($1->id_attr == SMNULL)
	         warn1("Unknown module %s", $1->ident,308);
              $$ = make_global_entity($1, MODULE_NAME, global_default, NO);
	      module_scope = $$->entry.Template.func_hedr;
           
            }
        ;

only_list: only_name
	    { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	| only_list COMMA only_name
	    { $$ = set_ll_list($1, $3, EXPR_LIST); }
	;

only_name: rename_name
            {  $$ = $1; }
         | name
	    {  PTR_HASH oldhash,copyhash;
	       PTR_SYMB oldsym, newsym;
	       PTR_LLND m;

	       oldhash = just_look_up_sym_in_scope(module_scope, $1->ident);
	       if (oldhash == HSNULL) {
                  errstr("Unknown identifier %s.", $1->ident,309);
	          $$= LLNULL;
	       }
	       else {
                 oldsym = oldhash->id_attr;
                 copyhash=just_look_up_sym_in_scope(cur_scope(), $1->ident);
	         if( copyhash && copyhash->id_attr && copyhash->id_attr->entry.Template.tag==module_scope->id)
                 {
                   newsym = copyhash->id_attr;
                   newsym->entry.Template.tag = 0;
                 }
                 else
                 {
	           newsym = make_local_entity($1, oldsym->variant, oldsym->type,LOCAL);
	           /* copies data in entry.Template structure and attr */
	           copy_sym_data(oldsym, newsym);	         
	             /*newsym->entry.Template.base_name = oldsym;*//*19.03.03*/
                 }
	  	/* l = make_llnd(fi, VAR_REF, LLNULL, LLNULL, oldsym);*/
		 m = make_llnd(fi, VAR_REF, LLNULL, LLNULL, newsym);
		 $$ = make_llnd(fi, RENAME_NODE, m, LLNULL, oldsym);
 	      }
            }
	;


rename_list: rename_name
	    { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	| rename_list COMMA rename_name
	    { $$ = set_ll_list($1, $3, EXPR_LIST); }
	;

rename_name: name POINT_TO name
	    {  PTR_HASH oldhash,copyhash;
	       PTR_SYMB oldsym, newsym;
	       PTR_LLND l, m;

	       oldhash = just_look_up_sym_in_scope(module_scope, $3->ident);
	       if (oldhash == HSNULL) {
                  errstr("Unknown identifier %s", $3->ident,309);
	          $$= LLNULL;
	       }
	       else {
                 oldsym = oldhash->id_attr;
                 copyhash = just_look_up_sym_in_scope(cur_scope(), $3->ident);
	         if(copyhash && copyhash->id_attr && copyhash->id_attr->entry.Template.tag==module_scope->id)
                 {
                    delete_symbol(copyhash->id_attr);
                    copyhash->id_attr = SMNULL;
                 }
                   newsym = make_local_entity($1, oldsym->variant, oldsym->type, LOCAL);
	           /* copies data in entry.Template structure and attr */
	           copy_sym_data(oldsym, newsym);	
                         
	           /*newsym->entry.Template.base_name = oldsym;*//*19.03.03*/
	  	 l  = make_llnd(fi, VAR_REF, LLNULL, LLNULL, oldsym);
		 m  = make_llnd(fi, VAR_REF, LLNULL, LLNULL, newsym);
		 $$ = make_llnd(fi, RENAME_NODE, m, l, SMNULL);
 	      }
            }
            ;
	
 /*
in_param: inside
	    {
	      if(parstate > INDCL)
		dclerr("parameter statement out of order", SMNULL);
	    }
	;
*/
dims:
	    { ndim = 0;	explicit_shape = 1; $$ = LLNULL; }
	| LEFTPAR dimlist RIGHTPAR
	    { $$ = $2; }
	;

dimlist:   { ndim = 0; explicit_shape = 1;}   dim
	    {
	      $$ = make_llnd(fi,EXPR_LIST, $2, LLNULL, SMNULL);
	      $$->type = global_default;
	    }
	| dimlist COMMA dim
	    { $$ = set_ll_list($1, $3, EXPR_LIST); }
	;

dim:	  ubound
	    {
	      if(ndim == maxdim)
		err("Too many dimensions", 43);
	      else if(ndim < maxdim)
		$$ = $1;
	      ++ndim;
	    }
	| COLON
	    {
	      if(ndim == maxdim)
		err("Too many dimensions", 43);
	      else if(ndim < maxdim)
		$$ = make_llnd(fi, DDOT, LLNULL, LLNULL, SMNULL);
	      ++ndim;
              explicit_shape = 0;
	    } 
	| expr COLON
	    {
	      if(ndim == maxdim)
		err("Too many dimensions", 43);
	      else if(ndim < maxdim)
		$$ = make_llnd(fi,DDOT, $1, LLNULL, SMNULL);
	      ++ndim;
              explicit_shape = 0;
	    }
	| expr COLON ubound
	    {
	      if(ndim == maxdim)
		err("Too many dimensions", 43);
	      else if(ndim < maxdim)
		$$ = make_llnd(fi,DDOT, $1, $3, SMNULL);
	      ++ndim;
	    }
	;

ubound:	  ASTER
	    {
	      $$ = make_llnd(fi,STAR_RANGE, LLNULL, LLNULL, SMNULL);
	      $$->type = global_default;
              explicit_shape = 0;
	    }
	| expr
	;

labellist: label
	    { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	| labellist COMMA label
	    { $$ = set_ll_list($1, $3, EXPR_LIST); }
	;

label:	  INT_CONSTANT
	    {PTR_LABEL p;
	     p = make_label_node(fi,convci(yyleng, yytext));
	     p->scope = cur_scope();
	     $$ = make_llnd_label(fi,LABEL_REF, p);
	  }
	;

implicit: IMPLICIT in_dcl implist
	   { /*PTR_LLND l;*/

          /*   l = make_llnd(fi, EXPR_LIST, $3, LLNULL, SMNULL);*/
             $$ = get_bfnd(fi,IMPL_DECL, SMNULL, $3, LLNULL, LLNULL);
             redefine_func_arg_type();
           }
/*
	| implicit COMMA implist
	  { PTR_LLND l;

            l = make_llnd(fi, EXPR_LIST, $3, LLNULL, SMNULL);
            add_to_lowLevelList(l, $1->entry.Template.ll_ptr1);
          }
 */
	| IMPLICITNONE
	  { /*undeftype = YES;
	    setimpl(TYNULL, (int)'a', (int)'z'); FB COMMENTED---> NOT QUITE RIGHT BUT AVOID PB WITH COMMON*/
	    $$ = get_bfnd(fi,IMPL_DECL, SMNULL, LLNULL, LLNULL, LLNULL);
	  }
	;

implist: impitem
         { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
       | implist COMMA impitem
         { $$ = set_ll_list($1, $3, EXPR_LIST); }
       ;

impitem:  imptype LEFTPAR letgroups RIGHTPAR
          { 

            $$ = make_llnd(fi, IMPL_TYPE, $3, LLNULL, SMNULL);
            $$->type = vartype;
          }
/*	| imptype
	 {
           $$ = make_llnd(fi, IMPL_TYPE, LLNULL, LLNULL, SMNULL);
           $$->type = vartype;
         }
*/ /*30.10.03*/
	; 

/* The draft specification leads to big trouble. Check that up. */
/* For the time being. */
imptype:   { implkwd = YES; } type_implicit
	    { vartype = $2; }
	;

type_implicit: STAT typename
               { $$ = $2; }
             | type
               { $$ = $1;}
             ;

/* Not used. 
in_implicit: inside
	    {
	      if (parstate >= INDCL)
		dclerr("implicit statement out of order", SMNULL);
	    }
	;
*/
letgroups: letgroup
	    { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	| letgroups COMMA letgroup
	    { $$ = set_ll_list($1, $3, EXPR_LIST); }
	;

letgroup:  letter
	    {
	      setimpl(vartype, (int)$1, (int)$1);
	      $$ = make_llnd(fi,CHAR_VAL, LLNULL, LLNULL, SMNULL);
	      $$->entry.cval = $1;
	    }
	| letter MINUS letter
	    { PTR_LLND p,q;
	      
	      setimpl(vartype, (int)$1, (int)$3);
	      p = make_llnd(fi,CHAR_VAL, LLNULL, LLNULL, SMNULL);
	      p->entry.cval = $1;
	      q = make_llnd(fi,CHAR_VAL, LLNULL, LLNULL, SMNULL);
	      q->entry.cval = $3;
	      $$= make_llnd(fi,DDOT, p, q, SMNULL);
	    }
	;

letter:	 IDENTIFIER
	    {
	      if(yyleng!=1 || yytext[0]<'a' || yytext[0]>'z')
		{
		  err("IMPLICIT item must be single letter", 37);
		  $$ = '\0';
		}
	      else $$ = yytext[0];
	    }
	;

inside:
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
	  
	    }
	;

in_dcl:
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
        }

	;

opt_double_colon: 
	| COLON COLON
        ;         

/*
 * Grammar for expressions
 */

funarglist:
	    { $$ = LLNULL; endioctl(); }
	| funargs 
            { $$ = $1;  endioctl();}
	;

funarg:  expr
	    { $$ = $1; }
	| triplet
	    { $$ = $1; }
	| nameeq expr
	    { PTR_LLND l;
	      l = make_llnd(fi, KEYWORD_ARG, $1, $2, SMNULL);
	      l->type = $2->type;
              $$ = l; 
	    }

        ;



funargs:  in_ioctl  funarg
            { $$ = set_ll_list($2, LLNULL, EXPR_LIST);
              endioctl(); 
            }
        | funargs COMMA  in_ioctl  funarg
	    { $$ = set_ll_list($1, $4, EXPR_LIST);
              endioctl(); 
            }
        ;

subscript_list: expr
	    { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
        | subscript_list COMMA expr
	    { $$ = set_ll_list($1, $3, EXPR_LIST); }
        ;

expr:	  uexpr
            { $$ = $1; }
	| LEFTPAR expr RIGHTPAR
	    { $$ = $2; }
	| complex_const
	    { $$ = $1; }
	;

uexpr:	  lhs
            { $$ = $1; }
	| simple_const
            { $$ = $1; }
/*        | string_constant opt_substring
            { $$ = make_llnd(fi, ARRAY_OP, $1, $2, SMNULL); }
*/	| vec
            { $$ = $1; }
	| expr PLUS expr   %prec PLUS
	    { $$ = intrinsic_op_node("+", ADD_OP, $1, $3); }
	| expr MINUS expr   %prec PLUS
	    { $$ = intrinsic_op_node("-", SUBT_OP, $1, $3); }
	| expr ASTER expr
	    { $$ = intrinsic_op_node("*", MULT_OP, $1, $3); }
	| expr SLASH expr
	    { $$ = intrinsic_op_node("/", DIV_OP, $1, $3); }
	| expr DASTER expr
	    { $$ = intrinsic_op_node("**", EXP_OP, $1, $3); }      
        | defined_op expr %prec UNARY_OP
	      { $$ = defined_op_node($1, $2, LLNULL); }
	| PLUS expr  %prec ASTER
	    { $$ = intrinsic_op_node("+", UNARY_ADD_OP, $2, LLNULL); }
	| MINUS expr  %prec ASTER
	    { $$ = intrinsic_op_node("-", MINUS_OP, $2, LLNULL); }
	| expr EQ expr  %prec EQ
	    { $$ = intrinsic_op_node(".eq.", EQ_OP, $1, $3); }
	| expr GT expr  %prec EQ
	    { $$ = intrinsic_op_node(".gt.", GT_OP, $1, $3); }
	| expr LT expr  %prec EQ
	    { $$ = intrinsic_op_node(".lt.", LT_OP, $1, $3); }
	| expr GE expr  %prec EQ
	    { $$ = intrinsic_op_node(".ge.", GTEQL_OP, $1, $3); }
	| expr LE expr  %prec EQ
	    { $$ = intrinsic_op_node(".ge.", LTEQL_OP, $1, $3); }
	| expr NE expr  %prec EQ
	    { $$ = intrinsic_op_node(".ne.", NOTEQL_OP, $1, $3); }
	| expr EQV expr
	    { $$ = intrinsic_op_node(".eqv.", EQV_OP, $1, $3); }
	| expr NEQV expr
	    { $$ = intrinsic_op_node(".neqv.", NEQV_OP, $1, $3); }
	| expr XOR expr
	    { $$ = intrinsic_op_node(".xor.", XOR_OP, $1, $3); }
	| expr OR expr
	    { $$ = intrinsic_op_node(".or.", OR_OP, $1, $3); }
	| expr AND expr
	    { $$ = intrinsic_op_node(".and.", AND_OP, $1, $3); }
	| NOT expr
	    { $$ = intrinsic_op_node(".not.", NOT_OP, $2, LLNULL); }
	| expr DSLASH expr
	    { $$ = intrinsic_op_node("//", CONCAT_OP, $1, $3); }
	| expr defined_op expr %prec BINARY_OP
	    { $$ = defined_op_node($2, $1, $3); }
        ;

addop:	  PLUS	{ $$ = ADD_OP; }
	| MINUS	{ $$ = SUBT_OP; }
	;
/*
relop:	  EQ	{ $$ = EQ_OP; }
	| GT	{ $$ = GT_OP; }
	| LT	{ $$ = LT_OP; }
	| GE	{ $$ = GTEQL_OP; }
	| LE	{ $$ = LTEQL_OP; }
	| NE	{ $$ = NOTEQL_OP; }
	;
*/
ident: name
	    { PTR_SYMB s;
	      PTR_TYPE t;
	     /* PTR_LLND l;*/

       	      if (!(s = $1->id_attr))
              {
	         s = make_scalar($1, TYNULL, LOCAL);
	     	 s->decl = SOFT;
	      } 
	
	      switch (s->variant)
              {
	      case CONST_NAME:
		   $$ = make_llnd(fi,CONST_REF,LLNULL,LLNULL, s);
		   t = s->type;
	           if ((t != TYNULL) &&
                       ((t->variant == T_ARRAY) ||  (t->variant == T_STRING) ))
                                 $$->variant = ARRAY_REF;

                   $$->type = t;
	           break;
	      case DEFAULT:   /* if common region with same name has been
                                 declared. */
		   s = make_scalar($1, TYNULL, LOCAL);
	     	   s->decl = SOFT;

	      case VARIABLE_NAME:
                   $$ = make_llnd(fi,VAR_REF,LLNULL,LLNULL, s);
	           t = s->type;
	           if (t != TYNULL) {
                     if ((t->variant == T_ARRAY) ||  (t->variant == T_STRING) ||
                         ((t->variant == T_POINTER) && (t->entry.Template.base_type->variant == T_ARRAY) ) )
                         $$->variant = ARRAY_REF;

/*  	              if (t->variant == T_DERIVED_TYPE)
                         $$->variant = RECORD_REF; */
	           }
                   $$->type = t;
	           break;
	      case TYPE_NAME:
  	           $$ = make_llnd(fi,TYPE_REF,LLNULL,LLNULL, s);
	           $$->type = s->type;
	           break;
	      case INTERFACE_NAME:
  	           $$ = make_llnd(fi, INTERFACE_REF,LLNULL,LLNULL, s);
	           $$->type = s->type;
	           break;
              case FUNCTION_NAME:
                   if(isResultVar(s)) {
                     $$ = make_llnd(fi,VAR_REF,LLNULL,LLNULL, s);
	             t = s->type;
	             if (t != TYNULL) {
                       if ((t->variant == T_ARRAY) ||  (t->variant == T_STRING) ||
                         ((t->variant == T_POINTER) && (t->entry.Template.base_type->variant == T_ARRAY) ) )
                         $$->variant = ARRAY_REF;
	             }
                     $$->type = t;
	             break;
                   }                                        
	      default:
  	           $$ = make_llnd(fi,VAR_REF,LLNULL,LLNULL, s);
	           $$->type = s->type;
	           break;
	      }
             /* if ($$->variant == T_POINTER) {
	         l = $$;
	         $$ = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	         $$->type = l->type->entry.Template.base_type;
	      }
              */ /*11.02.03*/
           } 
        ;

lhs:      ident
            { PTR_SYMB  s;
	      $$ = $1; 
              s= $$->entry.Template.symbol;
              if ((($1->variant == VAR_REF) || ($1->variant == ARRAY_REF))  && (s->scope !=cur_scope()))  /*global_bfnd*/
              {
	          if(((s->variant == FUNCTION_NAME) && (!isResultVar(s))) || (s->variant == PROCEDURE_NAME) || (s->variant == ROUTINE_NAME))
                  { s = $$->entry.Template.symbol =  make_scalar(s->parent, TYNULL, LOCAL);
		    $$->type = s->type;  
		  }
              }
            }
	| structure_component
            { $$ = $1; }
        |  array_ele_substring_func_ref
            { $$ = $1; }
	;

array_ele_substring_func_ref:  ident LEFTPAR in_ioctl  funarglist RIGHTPAR
	    { int num_triplets;
	      PTR_SYMB s;  /*, sym;*/
	      /* PTR_LLND l; */
	      PTR_TYPE tp;
	      /* l = $1; */
	      s = $1->entry.Template.symbol;
            
	      /* Handle variable to function conversion. */
	      if (($1->variant == VAR_REF) && 
	          (((s->variant == VARIABLE_NAME) && (s->type) &&
                    (s->type->variant != T_ARRAY)) ||
  	            (s->variant == ROUTINE_NAME))) {
	        s = $1->entry.Template.symbol =  make_function(s->parent, TYNULL, LOCAL);
	        $1->variant = FUNC_CALL;
              }
	      if (($1->variant == VAR_REF) && (s->variant == FUNCTION_NAME)) { 
                if(isResultVar(s))
	          $1->variant = ARRAY_REF;
                else
                  $1->variant = FUNC_CALL;
              }
	      if (($1->variant == VAR_REF) && (s->variant == PROGRAM_NAME)) {
                 errstr("The name '%s' is invalid in this context",s->ident,285);
                 $1->variant = FUNC_CALL;
              }
              /* l = $1; */
	      num_triplets = is_array_section_ref($4);
	      switch ($1->variant)
              {
	      case TYPE_REF:
                   $1->variant = STRUCTURE_CONSTRUCTOR;                  
                   $1->entry.Template.ll_ptr1 = $4;
                   $$ = $1;
                   $$->type =  lookup_type(s->parent); 
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

                   $1->variant = FUNC_CALL;

	      case FUNC_CALL:
                   $1->entry.Template.ll_ptr1 = $4;
                   $$ = $1;
                   if(s->type) 
                     $$->type = s->type;
                   else
                     $$->type = global_default;
	           /*late_bind_if_needed($$);*/ /*podd 02.02.23*/
	           break;
	      case DEREF_OP:
              case ARRAY_REF:
	           /* array element */
	           if (num_triplets == 0) {
                       if ($4 == LLNULL) {
                           s = $1->entry.Template.symbol = make_function(s->parent, TYNULL, LOCAL);
                           s->entry.func_decl.num_output = 1;
                           $1->variant = FUNC_CALL;
                           $$ = $1;
                       } else if ($1->type->variant == T_STRING) {
                           PTR_LLND temp = $4;
                           int num_input = 0;

                           while (temp) {
                             ++num_input;
                             temp = temp->entry.Template.ll_ptr2;
                           }
                           $1->entry.Template.ll_ptr1 = $4;
                           s = $1->entry.Template.symbol = make_function(s->parent, TYNULL, LOCAL);
                           s->entry.func_decl.num_output = 1;
                           s->entry.func_decl.num_input = num_input;
                           $1->variant = FUNC_CALL;
                           $$ = $1;
                       } else {
       	                   $1->entry.Template.ll_ptr1 = $4;
	                   $$ = $1;
                           $$->type = $1->type->entry.ar_decl.base_type;
                       }
                   }
                   /* substring */
	           else if ((num_triplets == 1) && 
                            ($1->type->variant == T_STRING)) {
    	           /*
                     $1->entry.Template.ll_ptr1 = $4;
	             $$ = $1; $$->type = global_string;
                   */
	                  $$ = make_llnd(fi, 
			  ARRAY_OP, LLNULL, LLNULL, SMNULL);
    	                  $$->entry.Template.ll_ptr1 = $1;
       	                  $$->entry.Template.ll_ptr2 = $4->entry.Template.ll_ptr1;
	                  $$->type = global_string;
                   }           
                   /* array section */
                   else {
    	             $1->entry.Template.ll_ptr1 = $4;
	             $$ = $1; tp = make_type(fi, T_ARRAY);     /**18.03.17*/
                     tp->entry.ar_decl.base_type = $1->type->entry.ar_decl.base_type; /**18.03.17 $1->type */
	             tp->entry.ar_decl.num_dimensions = num_triplets;
	             $$->type = tp;
                   }
	           break;
	      default:
                    if($1->entry.Template.symbol)
                      errstr("Can't subscript %s",$1->entry.Template.symbol->ident, 44);
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
           }
        | ident LEFTPAR in_ioctl  funarglist RIGHTPAR substring
	    { int num_triplets;
	      PTR_SYMB s;
	      PTR_LLND l;

	      s = $1->entry.Template.symbol;
/*              if ($1->type->variant == T_POINTER) {
	         l = $1;
	         $1 = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	         $1->type = l->type->entry.Template.base_type;
	      } */
	      if (($1->type->variant != T_ARRAY) ||
                  ($1->type->entry.ar_decl.base_type->variant != T_STRING)) {
	         errstr("Can't take substring of %s", s->ident, 45);
              }
              else {
  	        num_triplets = is_array_section_ref($4);
	           /* array element */
                if (num_triplets == 0) {
                   $1->entry.Template.ll_ptr1 = $4;
                  /* $1->entry.Template.ll_ptr2 = $6;*/
	          /* $$ = $1;*/
                   l=$1;
                   /*$$->type = $1->type->entry.ar_decl.base_type;*/
                   l->type = global_string;  /**18.03.17* $1->type->entry.ar_decl.base_type;*/
                }
                /* array section */
                else {
    	           $1->entry.Template.ll_ptr1 = $4;
    	           /*$1->entry.Template.ll_ptr2 = $6;
	           $$ = $1; $$->type = make_type(fi, T_ARRAY);
                   $$->type->entry.ar_decl.base_type = $1->type;
	           $$->type->entry.ar_decl.num_dimensions = num_triplets;
                  */
                   l = $1; l->type = make_type(fi, T_ARRAY);
                   l->type->entry.ar_decl.base_type = global_string;   /**18.03.17* $1->type*/
	           l->type->entry.ar_decl.num_dimensions = num_triplets;
               }
                $$ = make_llnd(fi, ARRAY_OP, l, $6, SMNULL);
	        $$->type = l->type;
              
              /* if ($$->variant == T_POINTER) {
	          l = $$;
	          $$ = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	          $$->type = l->type->entry.Template.base_type;
	       }
               */  /*11.02.03*/
             }
             endioctl();
          }
        | structure_component LEFTPAR funarglist RIGHTPAR
           {  int num_triplets;
	      PTR_LLND l,l1,l2;
              PTR_TYPE tp;

         /*   if ($1->variant == T_POINTER) {
	         l = $1;
	         $1 = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	         $1->type = l->type->entry.Template.base_type;
	      } */

              num_triplets = is_array_section_ref($3);
              $$ = $1;
              l2 = $1->entry.Template.ll_ptr2;  
              l1 = $1->entry.Template.ll_ptr1;                
              if(l2 && l2->type->variant == T_STRING)/*substring*/
                if(num_triplets == 1){
	           l = make_llnd(fi, ARRAY_OP, LLNULL, LLNULL, SMNULL);
    	           l->entry.Template.ll_ptr1 = l2;
       	           l->entry.Template.ll_ptr2 = $3->entry.Template.ll_ptr1;
	           l->type = global_string; 
                   $$->entry.Template.ll_ptr2 = l;                                          
                } else
                   err("Can't subscript",44);
              else if (l2 && l2->type->variant == T_ARRAY) {
                 if(num_triplets > 0) { /*array section*/
                   tp = make_type(fi,T_ARRAY);
                   tp->entry.ar_decl.base_type = $1->type->entry.ar_decl.base_type;
                   tp->entry.ar_decl.num_dimensions = num_triplets;
                   $$->type = tp;
                   l2->entry.Template.ll_ptr1 = $3;
                   l2->type = $$->type;   
                  }                 
                 else {  /*array element*/
                   l2->type = l2->type->entry.ar_decl.base_type;
                   l2->entry.Template.ll_ptr1 = $3;   
                   if(l1->type->variant != T_ARRAY)  
                     $$->type = l2->type;
                 }
              } else 
                   {err("Can't subscript",44); /*fprintf(stderr,"%d  %d",$1->variant,l2);*/}
                   /*errstr("Can't subscript %s",l2->entry.Template.symbol->ident,441);*/
         }    
	
        | structure_component LEFTPAR funarglist RIGHTPAR substring
	    { int num_triplets;
	      PTR_LLND l,q;

          /*     if ($1->variant == T_POINTER) {
	         l = $1;
	         $1 = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	         $1->type = l->type->entry.Template.base_type;
	      } */

              $$ = $1;
	      if (($1->type->variant != T_ARRAY) &&
                  ($1->type->entry.ar_decl.base_type->variant != T_STRING)) {
	         err("Can't take substring",45);
              }
              else {
  	        num_triplets = is_array_section_ref($3);
                l = $1->entry.Template.ll_ptr2;
                if(l) {
                /* array element */
	        if (num_triplets == 0) {
                   l->entry.Template.ll_ptr1 = $3;       	           
                   l->type = global_string;
                }
                /* array section */
                else {	
    	             l->entry.Template.ll_ptr1 = $3;
	             l->type = make_type(fi, T_ARRAY);
                     l->type->entry.ar_decl.base_type = global_string;
	             l->type->entry.ar_decl.num_dimensions = num_triplets;
                }
	        q = make_llnd(fi, ARRAY_OP, l, $5, SMNULL);
	        q->type = l->type;
                $$->entry.Template.ll_ptr2 = q;
                if($1->entry.Template.ll_ptr1->type->variant != T_ARRAY)  
                     $$->type = q->type;
               }
             }
          }
	;


structure_component: lhs PERCENT IDENTIFIER
	    { PTR_TYPE t;
	      PTR_SYMB  field;
	    /*  PTR_BFND at_scope;*/
              PTR_LLND l;


/*              if ($1->variant == T_POINTER) {
	         l = $1;
	         $1 = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	         $1->type = l->type->entry.Template.base_type;
	      } */

	      t = $1->type; 
	      
	      if (( ( ($1->variant == VAR_REF) 
	          ||  ($1->variant == CONST_REF) 
                  ||  ($1->variant == ARRAY_REF)
                  ||  ($1->variant == RECORD_REF)) && (t->variant == T_DERIVED_TYPE)) 
	          ||((($1->variant == ARRAY_REF) || ($1->variant == RECORD_REF)) && (t->variant == T_ARRAY) &&
                      (t = t->entry.ar_decl.base_type) && (t->variant == T_DERIVED_TYPE))) 
                {
                 t->name = lookup_type_symbol(t->name);
	         if ((field = component(t->name, yytext))) {                   
	            l =  make_llnd(fi, VAR_REF, LLNULL, LLNULL, field);
                    l->type = field->type;
                    if(field->type->variant == T_ARRAY || field->type->variant == T_STRING)
                      l->variant = ARRAY_REF; 
                    $$ = make_llnd(fi, RECORD_REF, $1, l, SMNULL);
                    if($1->type->variant != T_ARRAY)
                       $$->type = field->type;
                    else {
                       $$->type = make_type(fi,T_ARRAY);
                       if(field->type->variant != T_ARRAY) 
	                 $$->type->entry.ar_decl.base_type = field->type;
                       else
                         $$->type->entry.ar_decl.base_type = field->type->entry.ar_decl.base_type;
	               $$->type->entry.ar_decl.num_dimensions = t->entry.ar_decl.num_dimensions;
                       }
                 }
                  else  
                    errstr("Illegal component  %s", yytext,311);
              }                     
               else 
                    errstr("Can't take component  %s", yytext,311);
             }
              /*   if ($$->variant == T_POINTER) {
	            l = $$;
	            $$ = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	            $$->type = l->type->entry.Template.base_type;
	         }
              }
                else errstr("Can't take component of %s", yytext, 311);
              */
	   
        ;


array_element: ident
              { $$ = $1;}
            | structure_component
              {$$ = $1;}
            | ident LEFTPAR in_ioctl funarglist RIGHTPAR
            {  int num_triplets;
               PTR_TYPE tp;
              /* PTR_LLND l;*/
	      if ($1->type->variant == T_ARRAY)
              {
  	         num_triplets = is_array_section_ref($4);
	         /* array element */
	         if (num_triplets == 0) {
       	            $1->entry.Template.ll_ptr1 = $4;
       	            $$ = $1;
                    $$->type = $1->type->entry.ar_decl.base_type;
                 }
                 /* substring */
	       /*  else if ((num_triplets == 1) && 
                          ($1->type->variant == T_STRING)) {
    	                  $1->entry.Template.ll_ptr1 = $4;
	                  $$ = $1; $$->type = global_string;
                 }   */ /*podd*/        
                 /* array section */
                 else {
    	             $1->entry.Template.ll_ptr1 = $4;
	             $$ = $1; tp = make_type(fi, T_ARRAY);
                     tp->entry.ar_decl.base_type = $1->type->entry.ar_decl.base_type;  /**18.03.17* $1->type */
	             tp->entry.ar_decl.num_dimensions = num_triplets;
                     $$->type = tp;
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
           }
        | structure_component LEFTPAR funarglist RIGHTPAR

           {  int num_triplets;
	      PTR_LLND l,l1,l2;

         /*   if ($1->variant == T_POINTER) {
	         l = $1;
	         $1 = make_llnd(fi, DEREF_OP, l, LLNULL, SMNULL);
	         $1->type = l->type->entry.Template.base_type;
	      } */

              num_triplets = is_array_section_ref($3);
              $$ = $1;
              l2 = $1->entry.Template.ll_ptr2;  
              l1 = $1->entry.Template.ll_ptr1;                
              if(l2 && l2->type->variant == T_STRING)/*substring*/
                if(num_triplets == 1){
	           l = make_llnd(fi, ARRAY_OP, LLNULL, LLNULL, SMNULL);
    	           l->entry.Template.ll_ptr1 = l2;
       	           l->entry.Template.ll_ptr2 = $3->entry.Template.ll_ptr1;
	           l->type = global_string; 
                   $$->entry.Template.ll_ptr2 = l;                                          
                } else
                   err("Can't subscript",44);
              else if (l2 && l2->type->variant == T_ARRAY) {
                 if(num_triplets > 0) { /*array section*/
                   $$->type = make_type(fi,T_ARRAY);
                   $$->type->entry.ar_decl.base_type = l2->type->entry.ar_decl.base_type;
                   $$->type->entry.ar_decl.num_dimensions = num_triplets;
                   l2->entry.Template.ll_ptr1 = $3;
                   l2->type = $$->type;   
                  }                 
                 else {  /*array element*/
                   l2->type = l2->type->entry.ar_decl.base_type;
                   l2->entry.Template.ll_ptr1 = $3;   
                   if(l1->type->variant != T_ARRAY)  
                     $$->type = l2->type;
                 }
              } else 
                   err("Can't subscript",44);
         }  
         ;

asubstring: ident substring
            { 
	      if ($1->type->variant == T_STRING) {
                 $1->entry.Template.ll_ptr1 = $2;
                 $$ = $1; $$->type = global_string;
              }
              else errstr("can't subscript of %s", $1->entry.Template.symbol->ident,44);
            }
        ;

opt_substring:
            { $$ = LLNULL; }
        | substring
            { $$ = $1; }
        ;

substring:  LEFTPAR opt_expr COLON opt_expr RIGHTPAR
	    { $$ = make_llnd(fi, DDOT, $2, $4, SMNULL); }
	;

opt_expr:
	    { $$ = LLNULL; }
	| expr
	    { $$ = $1; }
	;

simple_const: numeric_bool_const
             { $$ = $1;}
            | numeric_bool_const UNDER kind
             { PTR_TYPE t;
               t = make_type_node($1->type, $3);
               $$ = $1;
               $$->type = t;
             }
            | integer_constant
	     { $$ = $1; }
            | integer_constant UNDER kind
             { PTR_TYPE t;
               t = make_type_node($1->type, $3);
               $$ = $1;
               $$->type = t;
             }
            | string_constant opt_substring
             {
              if ($2 != LLNULL)
              {
		 $$ = make_llnd(fi, ARRAY_OP, $1, $2, SMNULL); 
                 $$->type = global_string;
              }
	      else 
                 $$ = $1;
             }
            ;

numeric_bool_const:
	  TTRUE
	    {
	      $$ = make_llnd(fi,BOOL_VAL, LLNULL, LLNULL, SMNULL);
	      $$->entry.bval = 1;
	      $$->type = global_bool;
	    }
	| FFALSE
	    {
	      $$ = make_llnd(fi,BOOL_VAL, LLNULL, LLNULL, SMNULL);
	      $$->entry.bval = 0;
	      $$->type = global_bool;
	    }

	| REAL_CONSTANT
	    {
	      $$ = make_llnd(fi,FLOAT_VAL, LLNULL, LLNULL, SMNULL);
	      $$->entry.string_val = copys(yytext);
	      $$->type = global_float;
	    }
	| DP_CONSTANT
	    {
	      $$ = make_llnd(fi,DOUBLE_VAL, LLNULL, LLNULL, SMNULL);
	      $$->entry.string_val = copys(yytext);
	      $$->type = global_double;
	    }
	;

integer_constant: INT_CONSTANT
	    {
	      $$ = make_llnd(fi,INT_VAL, LLNULL, LLNULL, SMNULL);
	      $$->entry.ival = atoi(yytext);
	      $$->type = global_int;
	    }
                ;

string_constant: CHAR_CONSTANT
	    { PTR_TYPE t;
	      PTR_LLND p,q;
	      $$ = make_llnd(fi,STRING_VAL, LLNULL, LLNULL, SMNULL);
	      $$->entry.string_val = copys(yytext);
              if(yyquote=='\"') 
	        t = global_string_2;
              else
	        t = global_string;

	      p = make_llnd(fi,INT_VAL, LLNULL, LLNULL, SMNULL);
	      p->entry.ival = yyleng;
	      p->type = global_int;
              q = make_llnd(fi, LEN_OP, p, LLNULL, SMNULL); 
              $$->type = make_type_node(t, q);
	    }
              | ident  UNDER  CHAR_CONSTANT
            { PTR_TYPE t;
	      $$ = make_llnd(fi,STRING_VAL, LLNULL, LLNULL, SMNULL);
	      $$->entry.string_val = copys(yytext);
              if(yyquote=='\"') 
	        t = global_string_2;
              else
	        t = global_string;
	      $$->type = make_type_node(t, $1);
            }
              | integer_constant  UNDER  CHAR_CONSTANT
            { PTR_TYPE t;
	      $$ = make_llnd(fi,STRING_VAL, LLNULL, LLNULL, SMNULL);
	      $$->entry.string_val = copys(yytext);
              if(yyquote=='\"') 
	        t = global_string_2;
              else
	        t = global_string;
	      $$->type = make_type_node(t, $1);
            }
        ;


complex_const:	LEFTPAR uexpr COMMA uexpr RIGHTPAR
	    {
	      $$ = make_llnd(fi,COMPLEX_VAL, $2, $4, SMNULL);
	      $$->type = global_complex;
	    }
	;

kind:  ident
     { $$ = $1;}
    |  integer_constant
       { $$ = $1; }
   ;

/*
section: LEFTPAR triplets RIGHTPAR
	    { $$ = $2; }
	;

triplets:triplet
	    { $$ = make_llnd(fi,EXPR_LIST, $1, LLNULL, SMNULL); }

	| triplets COMMA triplet
	    { $$ = set_ll_list($1, $3, EXPR_LIST); }*/
/* I changed uexpr to expr in below rule and triplet rules.
    Don't know why Daya wrote it as uexpr. Srinivas. */
/*	| triplets COMMA expr
	    { $$ = set_ll_list($1, $3, EXPR_LIST); } */
/* Deleted. Purpose unknown. Srinivas. 
	| expr COMMA triplets
	    { $$ = set_ll_list($1, $3, EXPR_LIST); } 
	;*/

triplet:  expr COLON expr
	    { $$ = make_llnd(fi,DDOT,$1,$3,SMNULL); }
        | expr COLON 
	    { $$ = make_llnd(fi,DDOT,$1,LLNULL,SMNULL); }
	| expr COLON expr COLON expr
	    { $$ = make_llnd(fi,DDOT,make_llnd(fi,DDOT,$1,$3,SMNULL),$5,SMNULL); }
        | expr COLON COLON expr
	    { $$ = make_llnd(fi,DDOT,make_llnd(fi,DDOT,$1,LLNULL,SMNULL),$4,SMNULL); }
	| COLON expr COLON expr
	    { $$ = make_llnd(fi,DDOT, make_llnd(fi,DDOT,LLNULL,$2,SMNULL),$4,SMNULL); }
	| COLON COLON expr
	    { $$ = make_llnd(fi,DDOT,make_llnd(fi,DDOT,LLNULL,LLNULL,SMNULL),$3,SMNULL); }
	| COLON expr
	    { $$ = make_llnd(fi,DDOT,LLNULL,$2,SMNULL); }
	| COLON
	    { $$ = make_llnd(fi,DDOT,LLNULL,LLNULL,SMNULL); }
	;

vec:	  LEFTAB end_ioctl {in_vec=YES;} outlist {in_vec=NO;} RIGHTAB
           { PTR_TYPE array_type;
             $$ = make_llnd (fi,CONSTRUCTOR_REF,$4,LLNULL,SMNULL); 
             /*$$->type = $2->type;*/ /*28.02.03*/
             array_type = make_type(fi, T_ARRAY);
	     array_type->entry.ar_decl.num_dimensions = 1;
             if($4->type->variant == T_ARRAY)
	       array_type->entry.ar_decl.base_type = $4->type->entry.ar_decl.base_type;
             else
               array_type->entry.ar_decl.base_type = $4->type;
             $$->type = array_type;
           }
	;
  
allocate_object: ident
	   { $$ = $1; }
	| structure_component
	   { $$ = $1; }
	;
/*
allocation_list: ident 
	   { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	| structure_component
	   { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }

allocation_list: allocate_object
	   { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
*/


/*allocation_list: array_element
	        { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	       | allocation_list COMMA  opt_key_word  array_element
	        { $$ = set_ll_list($1, $4, EXPR_LIST); opt_kwd_ = NO; }
               | allocation_list COMMA  opt_key_word STAT EQUAL ident
                { stat_alloc = $6; }
	;
*/

allocation_list: array_element
	        { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	       | allocation_list COMMA  in_ioctl  array_element 
	        { $$ = set_ll_list($1, $4, EXPR_LIST); endioctl(); }
               | allocation_list COMMA  in_ioctl nameeq ident 
                { stat_alloc = make_llnd(fi, SPEC_PAIR, $4, $5, SMNULL);
                  endioctl();
                }
	;


/*allocate_object_list: allocate_object 
	   { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	| allocate_object_list COMMA opt_key_word allocate_object 
	   { $$ = set_ll_list($1, $4, EXPR_LIST); opt_kwd_ = NO; }
       | allocate_object_list COMMA  opt_key_word STAT EQUAL ident
	   { stat_alloc = $6; }
	;
*/

allocate_object_list: allocate_object 
	   { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	| allocate_object_list COMMA in_ioctl allocate_object 
	   { $$ = set_ll_list($1, $4, EXPR_LIST); endioctl(); }
       | allocate_object_list COMMA  in_ioctl nameeq ident
	   { stat_alloc = make_llnd(fi, SPEC_PAIR, $4, $5, SMNULL);
             endioctl();
           }
	;
/*
opt_stat_spec: 
	   { $$ = LLNULL; }
	| COMMA needkeyword STAT EQUAL ident
	   { $$ = $5; }
	;
*/

stat_spec: 
           {stat_alloc = LLNULL;}
         ;

pointer_name_list: ident
	   { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	| pointer_name_list COMMA ident
	   { $$ = set_ll_list($1, $3, EXPR_LIST); }
	;

/*
 * Grammar for executable statements
 */

exec:	  iffable
	    { $$ = $1; }
	| whereable
            { $$ = $1; }
        | plain_do
            { $$ = $1; }
        | construct_name_colon  plain_do
	    {
              $$ = $2;
              $$->entry.Template.ll_ptr3 = $1;
            }
/*

	| PLAINDO end_spec intonlyon dotarget intonlyoff opt_comma do_var EQUAL expr COMMA expr
	    {
              if(!$4){
                 err("No label in DO statement"); 
                 $$ = BFNULL;
              }
              else {
               if( $4->labdefined)
		  execerr("no backward DO loops", (char *)NULL);
	        $$ = make_do(FOR_NODE, $4, $7, $9, $11, NULL);
	      }
             
	    }
	| PLAINDO end_spec intonlyon dotarget intonlyoff opt_comma do_var EQUAL expr COMMA expr COMMA expr
	    {
              if(!$4){
                 err("No label in DO statement"); 
                 $$ = BFNULL;
              }
              else {
               if( $4->labdefined)
		  execerr("no backward DO loops", (char *)NULL);
	        $$ = make_do(FOR_NODE, $4, $7, $9, $11, $13);
	      }
             
	    }
*/
/* PROCESSDO added for FORTRAN M 
	| opt_construct_name_colon PROCESSDO end_spec intonlyon dotarget intonlyoff opt_comma do_var EQUAL expr COMMA expr
	    {
              if( !do_name_err ) {
                if($5 && $5->labdefined)
                  err("No backward DO loops", 46);
                $$ = make_processdo(PROCESS_DO_STAT, $5, $8, $10, $12, NULL);
	        $$->entry.Template.ll_ptr3 = $1;
              }
            }
	| opt_construct_name_colon PROCESSDO end_spec intonlyon dotarget intonlyoff opt_comma do_var EQUAL expr COMMA expr COMMA expr
            {
              if( !do_name_err ) {
                if($5 && $5->labdefined)
                  err("No backward DO loops", 46);
                $$ = make_processdo(PROCESS_DO_STAT, $5, $8, $10, $12, $14);
	        $$->entry.Template.ll_ptr3 = $1;
              }
            }
*/
	| ENDUNIT end_spec opt_unit_name
	    { PTR_BFND biff;

	      $$ = get_bfnd(fi,CONTROL_END, SMNULL, LLNULL, LLNULL, LLNULL); 
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
              if ($$)
                $$->control_parent = biff;
              delete_beyond_scope_level(pred_bfnd);
            }
	| ENDDO end_spec opt_construct_name                  
/*          { $$ = get_bfnd(fi,CONTROL_END, SMNULL, LLNULL, LLNULL, LLNULL);}*/
/*          Had to be changed to accomodate PCF's do loops.  */
	    {
              make_extend($3);
              $$ = BFNULL; 
              /* delete_beyond_scope_level(pred_bfnd); */
             } 
 
/* ENDPROCESSDO was added by C.Y.Chen 
        | ENDPROCESSDO end_spec
            { $$ = make_enddoall(); }
*/
	/* end module is bit different. So shouldn't have end_spec. */
	/* It may be better to move it to spec. */
	| ENDMODULE opt_unit_name
          { $$ = get_bfnd(fi,CONTROL_END, SMNULL, LLNULL, LLNULL, LLNULL); 
	    bind(); 
	    delete_beyond_scope_level(pred_bfnd);
	    position = IN_OUTSIDE;
          }

/*
	| opt_construct_name_colon DOWHILE end_spec  needkeyword intonlyon dotarget intonlyoff WHILE LEFTPAR expr RIGHTPAR */
        | do_while
            { $$ = $1; }

        | construct_name_colon do_while
	    {
              $$ = $2;
              $$->entry.Template.ll_ptr3 = $1;
            }

/*	| CDOALL do_var EQUAL expr COMMA expr
	        {$$ = make_do(CDOALL_NODE, NULL, $2, $4, $6, NULL);
                 add_scope_level($$, NO);
                }
	| CDOALL do_var EQUAL expr COMMA expr COMMA expr
	        {$$ = make_do(CDOALL_NODE, NULL, $2, $4, $6, $8);
                 add_scope_level($$, NO);
                }
	| SDOALL do_var EQUAL expr COMMA expr
	        {$$ = make_do(SDOALL_NODE, NULL, $2, $4, $6, NULL);
                 add_scope_level($$, NO);
                }
	| SDOALL do_var EQUAL expr COMMA expr COMMA expr
	        {$$ = make_do(SDOALL_NODE, NULL, $2, $4, $6, $8);
                 add_scope_level($$, NO);
                }
	| DOACROSS do_var EQUAL expr COMMA expr
	        {$$ = make_do(DOACROSS_NODE, NULL, $2, $4, $6, NULL);
                 add_scope_level($$, NO);
                }
	| DOACROSS do_var EQUAL expr COMMA expr COMMA expr
	        {$$ = make_do(DOACROSS_NODE, NULL, $2, $4, $6, $8);
                 add_scope_level($$, NO);
                }
	| CDOACROSS do_var EQUAL expr COMMA expr
	        {$$ = make_do(CDOACROSS_NODE, NULL, $2, $4, $6, NULL);
                 add_scope_level($$, NO);
                }
	| CDOACROSS do_var EQUAL expr COMMA expr COMMA expr
	        {$$ = make_do(CDOACROSS_NODE, NULL, $2, $4, $6, $8);
                 add_scope_level($$, NO);
                }
	| LOOP
          {
	   make_loop();
	   $$ = BFNULL;
	  }

	| enddoall
	    { $$ = make_enddoall();
              delete_beyond_scope_level(pred_bfnd);
            }
*/

	| logif iffable
	    { thiswasbranch = NO;
              $1->variant = LOGIF_NODE;
              $$ = make_logif($1, $2);
	      set_blobs($1, pred_bfnd, SAME_GROUP);
	    }
	| logif THEN
	    {
              $$ = $1;
	      set_blobs($$, pred_bfnd, NEW_GROUP1); 
            }
	|  construct_name_colon logif THEN
	    {
              $$ = $2;
              $$->entry.Template.ll_ptr3 = $1;
	      set_blobs($$, pred_bfnd, NEW_GROUP1); 
            }
	 /* | CONSTRUCT_ID COLON logif THEN
	    { PTR_SYMB s;
	      PTR_LLND l;
              
	      s = make_local_entity(look_up_sym(yytext), CONSTRUCT_NAME, global_default, LOCAL);
              l = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
	      * $$ = make_if($3); *
	      $3->entry.Template.ll_ptr3 = l;
              $$ = $3;
	      set_blobs($$, pred_bfnd, NEW_GROUP1); 
            }
         */
	| ELSEIF end_spec LEFTPAR expr RIGHTPAR THEN opt_construct_name
	    { make_elseif($4,$7); lastwasbranch = NO; $$ = BFNULL;}
	| ELSE end_spec opt_construct_name
	    { make_else($3); lastwasbranch = NO; $$ = BFNULL; }
	| ENDIF end_spec opt_construct_name
	    { make_endif($3); $$ = BFNULL; }
	| case
	    { $$ = $1; }
	| CONTAINS end_spec
	    { $$ = get_bfnd(fi, CONTAINS_STMT, SMNULL, LLNULL, LLNULL, LLNULL); }
        
        | forall iffable
	    { thiswasbranch = NO;
              $1->variant = FORALL_STAT;
              $$ = make_logif($1, $2);
	      set_blobs($1, pred_bfnd, SAME_GROUP);
	    }
        | forall
            { $$ = $1; }
        | construct_name_colon  forall
            { $$ = $2; $$->entry.Template.ll_ptr3 = $1;}
        | ENDFORALL end_spec  opt_construct_name
              { make_endforall($3); $$ = BFNULL; }   
               
        | dvm_exec
            { $$ = $1; }
        | acc_directive
            { $$ = $1; }
        | spf_directive
            { $$ = $1; }                
	;
/*
do_while:  DOWHILE end_spec  LEFTPAR expr RIGHTPAR
          { 	     
	
	       $$ = make_do(WHILE_NODE, LBNULL, SMNULL, $4, LLNULL, LLNULL);
	          
           }
        | DOWHILE end_spec  intonlyon dotarget intonlyoff needkeyword WHILE LEFTPAR expr RIGHTPAR 
        
          { 	     
	       if($4 && $4->labdefined)
		 execerr("no backward DO loops", (char *)NULL); 
	       $$ = make_do(WHILE_NODE, $4, SMNULL, $9, LLNULL, LLNULL);
	       	     
           }
        | DOWHILE end_spec intonlyon dotarget intonlyoff 
            {
               if( $4 && $4->labdefined)
		  err("No backward DO loops", 46);
	        $$ = make_do(WHILE_NODE, $4, SMNULL, LLNULL, LLNULL, LLNULL);            
	    }
     ;
*/

do_while:  DOWHILE end_spec  LEFTPAR expr RIGHTPAR
          { 	     
	     /*  if($5 && $5->labdefined)
		 execerr("no backward DO loops", (char *)NULL); */
	       $$ = make_do(WHILE_NODE, LBNULL, SMNULL, $4, LLNULL, LLNULL);
	       /*$$->entry.Template.ll_ptr3 = $1;*/	     
           }

  
       | DOWHILE end_spec intonlyon dotarget intonlyoff  opt_key_word opt_while
            {
               if( $4 && $4->labdefined)
		  err("No backward DO loops", 46);
	        $$ = make_do(WHILE_NODE, $4, SMNULL, $7, LLNULL, LLNULL);            
	    }
     ;

opt_while: 
	    { $$ = LLNULL; }
           |  COMMA needkeyword WHILE LEFTPAR expr RIGHTPAR
            { $$ = $5;}
	   |  WHILE LEFTPAR expr RIGHTPAR 
	    { $$ = $3;}
            
	;

plain_do: PLAINDO end_spec intonlyon dotarget intonlyoff opt_comma do_var EQUAL expr COMMA expr
	    {  
               if( $4 && $4->labdefined)
		  err("No backward DO loops", 46);
	        $$ = make_do(FOR_NODE, $4, $7, $9, $11, LLNULL);            
	    }

	| PLAINDO end_spec intonlyon dotarget intonlyoff opt_comma do_var EQUAL expr COMMA expr COMMA expr
	    {
               if( $4 && $4->labdefined)
		  err("No backward DO loops", 46);
	        $$ = make_do(FOR_NODE, $4, $7, $9, $11, $13);            
	    }
        ;

case:	CASE end_spec case_selector opt_construct_name
	    { $$ = get_bfnd(fi, CASE_NODE, $4, $3, LLNULL, LLNULL); }
      | DEFAULT_CASE  end_spec opt_construct_name
	    { /*PTR_LLND p;*/
	     /* p = make_llnd(fi, DEFAULT, LLNULL, LLNULL, SMNULL); */
	      $$ = get_bfnd(fi, DEFAULT_NODE, $3, LLNULL, LLNULL, LLNULL); }
      | ENDSELECT  end_spec opt_construct_name
        { make_endselect($3); $$ = BFNULL; }
	   /* { $$ = get_bfnd(fi, CONTROL_END, $3, LLNULL, LLNULL, LLNULL); }*/
      | SELECT  end_spec needkeyword CASE LEFTPAR expr RIGHTPAR
	    { $$ = get_bfnd(fi, SWITCH_NODE, SMNULL, $6, LLNULL, LLNULL) ; }
      | construct_name_colon SELECT  end_spec needkeyword CASE LEFTPAR expr RIGHTPAR
	    { $$ = get_bfnd(fi, SWITCH_NODE, SMNULL, $7, LLNULL, $1) ; }
	;

case_selector: LEFTPAR case_value_range_list RIGHTPAR
	         { $$ = $2; }
/*	| DEFAULT_CASE
	    { $$ = make_llnd(fi, DEFAULT, LLNULL, LLNULL, SMNULL); }
*/	;

case_value_range: expr
	            { $$ = $1; }
	| expr COLON
	    { $$ = make_llnd(fi, DDOT, $1, LLNULL, SMNULL); }
	| COLON expr
	    { $$ = make_llnd(fi, DDOT, LLNULL, $2, SMNULL); }
	| expr COLON expr
	    { $$ = make_llnd(fi, DDOT, $1, $3, SMNULL); }
	;

case_value_range_list: case_value_range
	                 { $$ = make_llnd(fi, EXPR_LIST, $1, LLNULL, SMNULL); }
	| case_value_range_list COMMA case_value_range
	    { PTR_LLND p;
	      
	      p = make_llnd(fi, EXPR_LIST, $3, LLNULL, SMNULL);
	      add_to_lowLevelList(p, $1);
	    }
	;

opt_construct_name:
	    { $$ = SMNULL; }
	| name
	    { $$ = make_local_entity($1, CONSTRUCT_NAME, global_default,
                                     LOCAL); } 

	;

opt_unit_name: 
              {$$ = HSNULL;}
             | name
              { $$ = $1;}
             ;

construct_name: CONSTRUCT_ID
               {$$ = look_up_sym(yytext);}
              ;

construct_name_colon:  construct_name COLON
	           { PTR_SYMB s;
	             s = make_local_entity( $1, CONSTRUCT_NAME, global_default, LOCAL);             
                    $$ = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
                   }
	           ;
/*
opt_construct_name_colon: 
	           { $$ = LLNULL; }
	          | construct_name COLON
	           { PTR_SYMB s;
	             s = make_local_entity( $1, CONSTRUCT_NAME, global_default, LOCAL);             
                    $$ = make_llnd(fi, VAR_REF, LLNULL, LLNULL, s);
                   }
	           ;
*/
/*
optkeyword: 
           {opt_kwd_ = YES;}
          ;
*/
logif:	  LOGICALIF end_spec LEFTPAR expr RIGHTPAR
	    { $$ = make_if($4); }
	;
forall:	  FORALL end_spec LEFTPAR forall_list opt_forall_cond RIGHTPAR
	    { $$ = make_forall($4,$5); }
	;

forall_list: forall_expr
             { $$ = make_llnd(fi, EXPR_LIST, $1, LLNULL, SMNULL); }
            | forall_list COMMA forall_expr
	    { PTR_LLND p;	      
	      p = make_llnd(fi, EXPR_LIST, $3, LLNULL, SMNULL);
	      add_to_lowLevelList(p, $1);
	    }
            ;

forall_expr: do_var EQUAL triplet
              {$$ = make_llnd(fi, FORALL_OP, $3, LLNULL, $1); }
           ;

opt_forall_cond:
                 { $$=LLNULL;}
               |  COMMA expr
                 { $$=$2;}           
               ;
/*
enddoall: ENDCDOALL
        | ENDSDOALL
	| ENDDOACROSS
	| ENDCDOACROSS
	;
*/

do_var:	 name
         { PTR_SYMB  s;
              s = $1->id_attr;
      	      if (!s || s->variant == DEFAULT)
              {
	         s = make_scalar($1, TYNULL, LOCAL);
	     	 s->decl = SOFT;
	      }
              $$ = s; 
	 }
	;

dospec:	  do_var EQUAL expr COMMA expr
             /* name EQUAL expr COMMA expr*/  /*16.02.03*/
	    { PTR_SYMB s;
              PTR_LLND l;
              int vrnt;

            /*  s = make_scalar($1, TYNULL, LOCAL);*/ /*16.02.03*/
              s = $1;
	      if (s->variant != CONST_NAME) {
                if(in_vec) 
                   vrnt=SEQ;
                else
                   vrnt=DDOT;     
                l = make_llnd(fi, SEQ, make_llnd(fi, vrnt, $3, $5, SMNULL),
                              LLNULL, SMNULL);
		$$ = make_llnd(fi,IOACCESS, LLNULL, l, s);
		do_name_err = NO;
	      }
	      else {
		err("Symbolic constant not allowed as DO variable", 47);
		do_name_err = YES;
	      }
	    }
	| do_var EQUAL expr COMMA expr COMMA expr
              /*name EQUAL expr COMMA expr COMMA expr*/ /*16.02.03*/
	    { PTR_SYMB s;
              PTR_LLND l;
              int vrnt;
              /*s = make_scalar($1, TYNULL, LOCAL);*/ /*16.02.03*/
              s = $1;
	      if( s->variant != CONST_NAME ) {
                if(in_vec) 
                   vrnt=SEQ;
                else
                   vrnt=DDOT;     
                l = make_llnd(fi, SEQ, make_llnd(fi, vrnt, $3, $5, SMNULL), $7,
                              SMNULL);
		$$ = make_llnd(fi,IOACCESS, LLNULL, l, s);
		do_name_err = NO;
	      }
	      else {
		err("Symbolic constant not allowed as DO variable", 47);
		do_name_err = YES;
	      }
	    }
	;


dotarget: { $$ = LBNULL; } 
        | INT_CONSTANT
            {
	       $$  = make_label_node(fi,convci(yyleng, yytext));
	       $$->scope = cur_scope();
	    }
       ;

whereable: ENDWHERE  end_spec opt_construct_name
           { make_endwhere($3); $$ = BFNULL; }
	 | ELSEWHERE  end_spec opt_construct_name
           { make_elsewhere($3); lastwasbranch = NO; $$ = BFNULL; }
	 | ELSEWHERE  end_spec LEFTPAR expr RIGHTPAR opt_construct_name
           { make_elsewhere_mask($4,$6); lastwasbranch = NO; $$ = BFNULL; }
	 | WHERE  end_spec LEFTPAR expr RIGHTPAR 
           { $$ = get_bfnd(fi, WHERE_BLOCK_STMT, SMNULL, $4, LLNULL, LLNULL); }
	 | construct_name_colon WHERE end_spec LEFTPAR expr RIGHTPAR 
           { $$ = get_bfnd(fi, WHERE_BLOCK_STMT, SMNULL, $5, LLNULL, $1); }
	 ;

iffable:  let expr EQUAL expr
          /* let lhs EQUAL expr */
           { PTR_LLND p, r;
             PTR_SYMB s1, s2 = SMNULL, s3, arg_list;
	     PTR_HASH hash_entry;

	   /*  if (just_look_up_sym("=") != HSNULL) {
	        p = intrinsic_op_node("=", EQUAL, $2, $4);
   	        $$ = get_bfnd(fi, OVERLOADED_ASSIGN_STAT, SMNULL, p, $2, $4);
             }	      
             else */ if ($2->variant == FUNC_CALL) {
                if(parstate==INEXEC){
                  	  err("Declaration among executables", 30);
                 /*   $$=BFNULL;*/
 	         $$ = get_bfnd(fi,STMTFN_STAT, SMNULL, $2, LLNULL, LLNULL);
                } 
                else {	         
  	         $2->variant = STMTFN_DECL;
		 /* $2->entry.Template.ll_ptr2 = $4; */
                 if( $2->entry.Template.ll_ptr1) {
		   r = $2->entry.Template.ll_ptr1->entry.Template.ll_ptr1;
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
 	         $$ = get_bfnd(fi,STMTFN_STAT, SMNULL, $2, LLNULL, LLNULL);
	         add_scope_level($$, NO);
                 arg_list = SMNULL;
		 if (s1) 
                 {
	            /*arg_list = SMNULL;*/
                    p = $2->entry.Template.ll_ptr1;
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
                       replace_symbol_in_expr(s3,$4);
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
  		    $2->entry.Template.ll_ptr1 = $4;
		    install_param_list($2->entry.Template.symbol,
				       arg_list, LLNULL, FUNCTION_NAME);
	            delete_beyond_scope_level($$);
		 
		/* else
		    errstr("cftn.gram: Illegal statement function declaration %s.", $2->entry.Template.symbol->ident); */
               }
	     }
	     else {
		$$ = get_bfnd(fi,ASSIGN_STAT,SMNULL, $2, $4, LLNULL);
                 parstate = INEXEC;
             }
	  }
	| POINTERLET end_spec lhs POINT_TO expr
	    { /*PTR_SYMB s;*/
	
	      /*s = make_scalar($2, TYNULL, LOCAL);*/
  	      $$ = get_bfnd(fi, POINTER_ASSIGN_STAT, SMNULL, $3, $5, LLNULL);
	    }
/*        | let lhs EQUAL vec
		
	       { if ($2->variant == ARRAY_REF)
		    $$ = get_bfnd(fi,ASSIGN_STAT,SMNULL, $2, $4, LLNULL);
                 else err("Constructor reference");
		}
*/	| ASSIGN end_spec label TO name
	    { PTR_SYMB p;

	      p = make_scalar($5, TYNULL, LOCAL);
	      p->variant = LABEL_VAR;
  	      $$ = get_bfnd(fi,ASSLAB_STAT, p, $3,LLNULL,LLNULL);
            }
	| CONTINUE end_spec
	    { $$ = get_bfnd(fi,CONT_STAT,SMNULL,LLNULL,LLNULL,LLNULL); }
	| goto
	| io
	    { inioctl = NO; }
	| ARITHIF end_spec LEFTPAR expr RIGHTPAR label COMMA label COMMA label
	    { PTR_LLND	p;

	      p = make_llnd(fi,EXPR_LIST, $10, LLNULL, SMNULL);
	      p = make_llnd(fi,EXPR_LIST, $8, p, SMNULL);
	      $$= get_bfnd(fi,ARITHIF_NODE, SMNULL, $4,
			    make_llnd(fi,EXPR_LIST, $6, p, SMNULL), LLNULL);
	      thiswasbranch = YES;
            }
	| call
	    {
	      $$ = subroutine_call($1, LLNULL, LLNULL, PLAIN);
/*	      match_parameters($1, LLNULL);
	      $$= get_bfnd(fi,PROC_STAT, $1, LLNULL, LLNULL, LLNULL);
*/	      endioctl(); 
            }
	| call LEFTPAR RIGHTPAR
	    {
	      $$ = subroutine_call($1, LLNULL, LLNULL, PLAIN);
/*	      match_parameters($1, LLNULL);
	      $$= get_bfnd(fi,PROC_STAT,$1,LLNULL,LLNULL,LLNULL);
*/	      endioctl(); 
	    }
	| call LEFTPAR  callarglist RIGHTPAR
	    {
	      $$ = subroutine_call($1, $3, LLNULL, PLAIN);
/*	      match_parameters($1, $3);
	      $$= get_bfnd(fi,PROC_STAT,$1,$3,LLNULL,LLNULL);
*/	      endioctl(); 
	    }

	| RETURN end_spec opt_expr
	    {
	      $$ = get_bfnd(fi,RETURN_STAT,SMNULL,$3,LLNULL,LLNULL);
	      thiswasbranch = YES;
	    }
	| stop end_spec opt_expr
	    {
	      $$ = get_bfnd(fi,$1,SMNULL,$3,LLNULL,LLNULL);
	      thiswasbranch = ($1 == STOP_STAT);
	    }
	| CYCLE  end_spec opt_construct_name
	    { $$ = get_bfnd(fi, CYCLE_STMT, $3, LLNULL, LLNULL, LLNULL); }

	| EXIT  end_spec opt_construct_name
	    { $$ = get_bfnd(fi, EXIT_STMT, $3, LLNULL, LLNULL, LLNULL); }

	| ALLOCATE  end_spec  LEFTPAR  stat_spec allocation_list RIGHTPAR
	    { $$ = get_bfnd(fi, ALLOCATE_STMT,  SMNULL, $5, stat_alloc, LLNULL); }

	| DEALLOCATE end_spec LEFTPAR  stat_spec allocate_object_list  RIGHTPAR
	    { $$ = get_bfnd(fi, DEALLOCATE_STMT, SMNULL, $5, stat_alloc , LLNULL); }

	| NULLIFY end_spec LEFTPAR pointer_name_list RIGHTPAR
	    { $$ = get_bfnd(fi, NULLIFY_STMT, SMNULL, $4, LLNULL, LLNULL); }

        | WHERE_ASSIGN end_spec LEFTPAR expr RIGHTPAR lhs EQUAL expr
           { $$ = get_bfnd(fi, WHERE_NODE, SMNULL, $4, $6, $8); }
	;

let:	  LET in_data
      /*
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
     }
     */
         {$$ = LLNULL;}
	;
 
goto:	  PLAINGOTO end_spec label
	    {
	      $$=get_bfnd(fi,GOTO_NODE,SMNULL,LLNULL,LLNULL,(PTR_LLND)$3);
	      thiswasbranch = YES;
	    }
	| ASSIGNGOTO end_spec name
	    { PTR_SYMB p;

	      if($3->id_attr)
		p = $3->id_attr;
	      else {
	        p = make_scalar($3, TYNULL, LOCAL);
		p->variant = LABEL_VAR;
	      }

	      if(p->variant == LABEL_VAR) {
		  $$ = get_bfnd(fi,ASSGOTO_NODE,p,LLNULL,LLNULL,LLNULL);
		  thiswasbranch = YES;
	      }
	      else {
		  err("Must go to assigned variable", 48);
		  $$ = BFNULL;
	      }
	    }
	| ASSIGNGOTO end_spec name opt_comma LEFTPAR labellist RIGHTPAR
	    { PTR_SYMB p;

	      if($3->id_attr)
		p = $3->id_attr;
	      else {
	        p = make_scalar($3, TYNULL, LOCAL);
		p->variant = LABEL_VAR;
	      }

	      if (p->variant == LABEL_VAR) {
		 $$ = get_bfnd(fi,ASSGOTO_NODE,p,$6,LLNULL,LLNULL);
		 thiswasbranch = YES;
	      }
	      else {
		err("Must go to assigned variable",48);
		$$ = BFNULL;
	      }
	    }
	| COMPGOTO end_spec LEFTPAR labellist RIGHTPAR opt_comma expr
	    { $$ = get_bfnd(fi,COMGOTO_NODE, SMNULL, $4, $7, LLNULL); }
	;

opt_comma:
	| COMMA
	;

call:	  CALL end_spec name in_ioctl
	    { $$ = make_procedure($3, LOCAL); }
	;

callarglist:  in_ioctl callarg
	    { 
              $$ = set_ll_list($2, LLNULL, EXPR_LIST);
              endioctl();
            }
	| callarglist COMMA in_ioctl callarg
	    { 
               $$ = set_ll_list($1, $4, EXPR_LIST);
               endioctl();
            }
	;

callarg:  expr
	    { $$ = $1; }
	| nameeq expr
	    { $$  = make_llnd(fi, KEYWORD_ARG, $1, $2, SMNULL); }
	| ASTER label
	    { $$ = make_llnd(fi,LABEL_ARG,$2,LLNULL,SMNULL); }
	;

stop:	  PAUSE	{ $$ = PAUSE_NODE; }
	| STOP	{ $$ = STOP_STAT; }
	;

/*
exprlist: expr
	    { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	| exprlist COMMA expr
	    { $$ = set_ll_list($1, $3, EXPR_LIST); }
	;
*/
end_spec:
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
	      }
	;

intonlyon:
	    { intonly = YES; }
	;

intonlyoff:
	     { intonly = NO; }

	;

/*
 *  Grammar for Input/Output statements
 */
io:	  iofmove ioctl
		{ $1->entry.Template.ll_ptr2 = $2;
		  $$ = $1; }
	| iofmove unpar_fexpr 
		{ PTR_LLND p, q = LLNULL;

		  q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  q->entry.string_val = (char *)"unit";
		  q->type = global_string;
		  p = make_llnd(fi, SPEC_PAIR, q, $2, SMNULL);
		  $1->entry.Template.ll_ptr2 = p;
		  endioctl();
		  $$ = $1; }
	| iofmove ASTER
		{ PTR_LLND p, q, r;

		  p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  p->entry.string_val = (char *)"*";
		  p->type = global_string;
		  q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  q->entry.string_val = (char *)"unit";
		  q->type = global_string;
		  r = make_llnd(fi, SPEC_PAIR, p, q, SMNULL);
		  $1->entry.Template.ll_ptr2 = r;
		  endioctl();
		  $$ = $1; }
 	| iofmove DASTER
		{ PTR_LLND p, q, r;

		  p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  p->entry.string_val = (char *)"**";
		  p->type = global_string;
		  q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  q->entry.string_val = (char *)"unit";
		  q->type = global_string;
		  r = make_llnd(fi, SPEC_PAIR, p, q, SMNULL);
		  $1->entry.Template.ll_ptr2 = r;
		  endioctl();
		  $$ = $1; }
	| iofctl ioctl
		{ $1->entry.Template.ll_ptr2 = $2;
		  $$ = $1; }
	| inquire
                { $$ = $1; }
	| read ioctl
		{ $1->entry.Template.ll_ptr2 = $2;
		  $$ = $1; }
	| read infmt
		{ $1->entry.Template.ll_ptr2 = $2;
		  $$ = $1; }
	| read ioctl inlist
		{ $1->entry.Template.ll_ptr2 = $2;
		  $1->entry.Template.ll_ptr1 = $3;
		  $$ = $1; }
	| read infmt COMMA inlist
		{ $1->entry.Template.ll_ptr2 = $2;
		  $1->entry.Template.ll_ptr1 = $4;
		  $$ = $1; }
/*	| read ioctl COMMA inlist                   Not needed
		{ $1->entry.Template.ll_ptr2 = $2;
		  $1->entry.Template.ll_ptr1 = $4;
		  $$ = $1; }  
*/
	| write ioctl
		{ $1->entry.Template.ll_ptr2 = $2;
		  $$ = $1; }
	| write ioctl outlist
		{ $1->entry.Template.ll_ptr2 = $2;
		  $1->entry.Template.ll_ptr1 = $3;
		  $$ = $1; }
	| print
		{ $$ = $1; }
	| print COMMA outlist
		{ $1->entry.Template.ll_ptr1 = $3;
		  $$ = $1; }
	;


iofmove:   fmkwd end_spec start_ioctl
		{ $$ = $1; }
	;

fmkwd:	  BACKSPACE
           {$$ = get_bfnd(fi, BACKSPACE_STAT, SMNULL, LLNULL, LLNULL, LLNULL);}
	| REWIND
           {$$ = get_bfnd(fi, REWIND_STAT, SMNULL, LLNULL, LLNULL, LLNULL);}
	| ENDFILE
           {$$ = get_bfnd(fi, ENDFILE_STAT, SMNULL, LLNULL, LLNULL, LLNULL);}
/*        | SKIPPASTEOF
           {$$ = get_bfnd(fi, SKIPPASTEOF_NODE, SMNULL, LLNULL, LLNULL, LLNULL); }
*/
	;

iofctl:  ctlkwd end_spec start_ioctl
		{ $$ = $1; }
	;

ctlkwd:   OPEN
           {$$ = get_bfnd(fi, OPEN_STAT, SMNULL, LLNULL, LLNULL, LLNULL);}
	| CLOSE
           {$$ = get_bfnd(fi, CLOSE_STAT, SMNULL, LLNULL, LLNULL, LLNULL);}
	;

inquire: INQUIRE end_spec start_ioctl ioctl
           {  $$ = get_bfnd(fi, INQUIRE_STAT, SMNULL, LLNULL, $4, LLNULL);}
        | INQUIRE end_spec start_ioctl  ioctl outlist
           {  $$ = get_bfnd(fi, INQUIRE_STAT, SMNULL, $5, $4, LLNULL);}
	;

infmt:	  unpar_fexpr
		{ PTR_LLND p;
		  PTR_LLND q = LLNULL;

		  if ($1->variant == INT_VAL)
 	          {
		        PTR_LABEL r;

			r = make_label_node(fi, (long) $1->entry.ival);
			r->scope = cur_scope();
			p = make_llnd_label(fi, LABEL_REF, r);
		  }
		  else p = $1; 
		  q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  q->entry.string_val = (char *)"fmt";
		  q->type = global_string;
		  $$ = make_llnd(fi, SPEC_PAIR, q, p, SMNULL);
		  endioctl();
		}
	| ASTER
		{ PTR_LLND p;
		  PTR_LLND q;

		  p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  p->entry.string_val = (char *)"*";
		  p->type = global_string;
		  q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  q->entry.string_val = (char *)"fmt";
		  q->type = global_string;
		  $$ = make_llnd(fi, SPEC_PAIR, q, p, SMNULL);
		  endioctl();
		}
	;


ioctl:	LEFTPAR fexpr RIGHTPAR
		{ PTR_LLND p;

		  p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  p->entry.string_val = (char *)"unit";
		  p->type = global_string;
		  $$ = make_llnd(fi, SPEC_PAIR, p, $2, SMNULL);
		  endioctl();
		}
	| LEFTPAR  ctllist RIGHTPAR

/*ioctl: LEFTPAR  ctllist RIGHTPAR*/
		{ 
		  $$ = $2;
		  endioctl();
		 }
	;

ctllist:  ioclause
		{ $$ = $1; endioctl();}
	| ctllist COMMA in_ioctl ioclause
		{ $$ = set_ll_list($1, $4, EXPR_LIST); endioctl();}
	;

ioclause:  fexpr
		{ PTR_LLND p;
		  PTR_LLND q;
 
		  nioctl++;
		  if ((nioctl == 2) && ($1->variant == INT_VAL))
 	          {
		        PTR_LABEL r;

			r = make_label_node(fi, (long) $1->entry.ival);
			r->scope = cur_scope();
			p = make_llnd_label(fi, LABEL_REF, r);
		  }
		  else p = $1; 
		  q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  if (nioctl == 1)
		        q->entry.string_val = (char *)"unit"; 
		  else {
                     if(($1->variant == VAR_REF) && $1->entry.Template.symbol->variant == NAMELIST_NAME)
                       q->entry.string_val = (char *)"nml";
                     else
                       q->entry.string_val = (char *)"fmt";
                  }
		  q->type = global_string;
		  $$ = make_llnd(fi, SPEC_PAIR, q, p, SMNULL);
		}
	| ASTER
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
		  $$ = make_llnd(fi, SPEC_PAIR, q, p, SMNULL);
		}
	| DASTER
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
		  $$ = make_llnd(fi, SPEC_PAIR, q, p, SMNULL);
		}
	| nameeq expr
		{ 
		  PTR_LLND p;
		  char *q;

		  q = $1->entry.string_val;
  		  if ((strcmp(q, "end") == 0) || (strcmp(q, "err") == 0) || (strcmp(q, "eor") == 0) || ((strcmp(q,"fmt") == 0) && ($2->variant == INT_VAL)))
 	          {
		        PTR_LABEL r;

			r = make_label_node(fi, (long) $2->entry.ival);
			r->scope = cur_scope();
			p = make_llnd_label(fi, LABEL_REF, r);
		  }
		  else p = $2;

		  $$ = make_llnd(fi, SPEC_PAIR, $1, p, SMNULL); }
	| nameeq ASTER
		{ PTR_LLND p;
                  
		  p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  p->entry.string_val = (char *)"*";
		  p->type = global_string;
		  $$ = make_llnd(fi, SPEC_PAIR, $1, p, SMNULL);
		}
	| nameeq DASTER
		{ PTR_LLND p;
		  p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  p->entry.string_val = (char *)"*";
		  p->type = global_string;
		  $$ = make_llnd(fi, SPEC_PAIR, $1, p, SMNULL);
		}
	;

nameeq:  NAMEEQ
		{ $$ = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
		  $$->entry.string_val = copys(yytext);
		  $$->type = global_string;
	        }
	;


read:	  READ end_spec start_ioctl
           {$$ = get_bfnd(fi, READ_STAT, SMNULL, LLNULL, LLNULL, LLNULL);}
	;


write:	  WRITE end_spec start_ioctl
           {$$ = get_bfnd(fi, WRITE_STAT, SMNULL, LLNULL, LLNULL, LLNULL);}
	;


print:	  PRINT end_spec fexpr start_ioctl
           {
	    PTR_LLND p, q, l;

	    if ($3->variant == INT_VAL)
		{
		        PTR_LABEL r;

			r = make_label_node(fi, (long) $3->entry.ival);
			r->scope = cur_scope();
			p = make_llnd_label(fi, LABEL_REF, r);
		}
	    else p = $3;
	    
            q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
	    q->entry.string_val = (char *)"fmt";
            q->type = global_string;
            l = make_llnd(fi, SPEC_PAIR, q, p, SMNULL);

            $$ = get_bfnd(fi, PRINT_STAT, SMNULL, LLNULL, l, LLNULL);
	    endioctl();
	   }	
	| PRINT end_spec ASTER start_ioctl
	   { PTR_LLND p, q, r;
		
	     p = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
	     p->entry.string_val = (char *)"*";
	     p->type = global_string;
	     q = make_llnd(fi, KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
	     q->entry.string_val = (char *)"fmt";
             q->type = global_string;
             r = make_llnd(fi, SPEC_PAIR, q, p, SMNULL);
	     $$ = get_bfnd(fi, PRINT_STAT, SMNULL, LLNULL, r, LLNULL);
	     endioctl();
           }
	;


inlist:	  inelt
		{ $$ = set_ll_list($1, LLNULL, EXPR_LIST);}
	| inlist COMMA inelt
		{ $$ = set_ll_list($1, $3, EXPR_LIST);}
	;

inelt:	  lhs
		{ $$ = $1; }
	| LEFTPAR inlist COMMA dospec RIGHTPAR
		{
		  $4->entry.Template.ll_ptr1 = $2;
		  $$ = $4;
		}
	;

outlist:  uexpr
		{ $$ = set_ll_list($1, LLNULL, EXPR_LIST);  $$->type = $1->type;}
	| other
		{ $$ = $1; }
	| out2
		{ $$ = $1; }
	;

out2:	  uexpr COMMA uexpr
		{ $$ = set_ll_list($1, $3, EXPR_LIST); $$->type = $1->type;}
	| uexpr COMMA other
		{ $$ = set_ll_list($1, $3, EXPR_LIST); $$->type = $1->type;}
	| other COMMA uexpr
		{ $$ = set_ll_list($1, $3, EXPR_LIST); $$->type = $1->type;}
	| other COMMA other
		{ $$ = set_ll_list($1, $3, EXPR_LIST); $$->type = $1->type;}
	| out2  COMMA uexpr
		{ $$ = set_ll_list($1, $3, EXPR_LIST); $$->type = $1->type;}
	| out2  COMMA other
		{ $$ = set_ll_list($1, $3, EXPR_LIST); $$->type = $1->type;}
	;

other:	  complex_const
		{ $$ =  set_ll_list($1, LLNULL, EXPR_LIST);
	          $$->type = global_complex; }
	| LEFTPAR expr RIGHTPAR
		{ $$ =  set_ll_list($2, LLNULL, EXPR_LIST);
                  $$->type = $2->type; }
	| LEFTPAR uexpr COMMA dospec RIGHTPAR
		{
		  $4->entry.Template.ll_ptr1 = $2;
		  $$ =  set_ll_list($4, LLNULL, EXPR_LIST);
                  $$->type = $2->type; 
		}
	| LEFTPAR other COMMA dospec RIGHTPAR	
		{
		  $4->entry.Template.ll_ptr1 = $2;
		  $$ =  set_ll_list($4, LLNULL, EXPR_LIST);
                  $$->type = $2->type; 
		}
	| LEFTPAR out2  COMMA dospec RIGHTPAR
		{
		  $4->entry.Template.ll_ptr1 = $2;
		  $$ =  set_ll_list($4, LLNULL, EXPR_LIST);
                  $$->type = $2->type; 
		}
	;

in_ioctl:
		{ inioctl = YES; }
	;

start_ioctl:
		{ startioctl();}
	;


/*
 * used by I/O statement 
 */
fexpr:	unpar_fexpr
            { $$ = $1; }
	| LEFTPAR fexpr RIGHTPAR
	    { $$ = $2; }
	;

unpar_fexpr:  lhs
        { $$ = $1; }
	| simple_const
        { $$ = $1; }
	| fexpr addop fexpr   %prec PLUS
	    {
	      $$ = make_llnd(fi,$2, $1, $3, SMNULL);
	      set_expr_type($$);
	    }
	| fexpr ASTER fexpr
	    {
	      $$ = make_llnd(fi,MULT_OP, $1, $3, SMNULL);
	      set_expr_type($$);
	    }
	| fexpr SLASH fexpr
	    {
	      $$ = make_llnd(fi,DIV_OP, $1, $3, SMNULL);
	      set_expr_type($$);
	    }
	| fexpr DASTER fexpr
	    {
	      $$ = make_llnd(fi,EXP_OP, $1, $3, SMNULL);
	      set_expr_type($$);
	    }
	| addop fexpr  %prec ASTER
	    {
	      if($1 == SUBT_OP)
		{
		  $$ = make_llnd(fi,SUBT_OP, $2, LLNULL, SMNULL);
		  set_expr_type($$);
		}
	      else	$$ = $2;
	    }
	| fexpr DSLASH fexpr
	    {
	      $$ = make_llnd(fi,CONCAT_OP, $1, $3, SMNULL);
	      set_expr_type($$);
	    }
        | IDENTIFIER EQUAL expr  
           { $$ = LLNULL; }
	;


cmnt:       /* nothing */
	  { comments = cur_comment = CMNULL; }
	| COMMENT
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
          }
	;
dvm_specification: dvm_template
                 | dvm_align  
                 | dvm_distribute   
                 | dvm_dynamic   
                 | dvm_processors 
                 | dvm_shadow
                 | dvm_combined_dir
                 | dvm_pointer 
                 | dvm_task
                 | dvm_inherit
                 | dvm_indirect_group
                 | dvm_remote_group
                 | dvm_reduction_group
                 | dvm_consistent_group
                 | dvm_heap
                 | dvm_asyncid
                 | dvm_consistent
                 | ompdvm_onethread /*OMP*/
		 | omp_specification_directive /*OMP*/
                 ; 

dvm_exec: dvm_redistribute
        | dvm_realign
        | dvm_parallel_on
        | dvm_shadow_group
        | dvm_shadow_start
        | dvm_shadow_wait
        | dvm_reduction_start
        | dvm_reduction_wait
        | dvm_consistent_start
        | dvm_consistent_wait
        | dvm_remote_access
        | dvm_task_region
        | dvm_end_task_region
        | dvm_map
        | dvm_on
        | dvm_end_on
        | dvm_indirect_access
        | dvm_prefetch
        | dvm_reset
        | dvm_debug_dir
        | dvm_enddebug_dir
        | dvm_interval_dir
        | dvm_endinterval_dir
        | dvm_exit_interval_dir
        | dvm_traceon_dir
        | dvm_traceoff_dir
        | dvm_barrier_dir
        | dvm_check
        | dvm_new_value   
        | dvm_asynchronous
        | dvm_endasynchronous
        | dvm_asyncwait
        | dvm_f90
        | dvm_io_mode_dir
        | dvm_shadow_add
        | dvm_localize
        | dvm_cp_create
        | dvm_cp_load
        | dvm_cp_save
        | dvm_cp_wait
        | dvm_template_create
        | dvm_template_delete
        | hpf_independent 
	| omp_execution_directive /*OMP*/
/*        | dvm_own      */
        ;
       
dvm_template: HPF_TEMPLATE in_dcl template_obj	    
	      { $$ = get_bfnd(fi,HPF_TEMPLATE_STAT, SMNULL, $3, LLNULL, LLNULL); }
	    | dvm_template COMMA template_obj 
              { PTR_SYMB s;
                if($1->entry.Template.ll_ptr2)
                {
                  s = $3->entry.Template.ll_ptr1->entry.Template.symbol;
                  s->attr = s->attr | COMMON_BIT;
                }
	        add_to_lowLevelList($3, $1->entry.Template.ll_ptr1);
	      }
	    ;

template_obj: name dims
             {PTR_SYMB s;
	      PTR_LLND q;
	    /* 27.06.18
	      if(! explicit_shape)   
                err("Explicit shape specification is required", 50);
	    */  
	      s = make_array($1, TYNULL, $2, ndim, LOCAL);
              if(s->attr & TEMPLATE_BIT)
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
              if((s->attr & PROCESSORS_BIT) || (s->attr & TASK_BIT) || (s->attr & DVM_POINTER_BIT))
                errstr( "Inconsistent declaration of identifier  %s ", s->ident, 16);
              else
	        s->attr = s->attr | TEMPLATE_BIT;
              if($2) s->attr = s->attr | DIMENSION_BIT;  
	      q = make_llnd(fi,ARRAY_REF, $2, LLNULL, s);
	      s->type->entry.ar_decl.ranges = $2;
	      $$ = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	     }
             ;

dvm_dynamic: DYNAMIC in_dcl dyn_array_name_list
    { $$ = get_bfnd(fi,DVM_DYNAMIC_DIR, SMNULL, $3, LLNULL, LLNULL);}
           ;

dyn_array_name_list: dyn_array_name
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
              | dyn_array_name_list COMMA dyn_array_name
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;

dyn_array_name: name
           {  PTR_SYMB s;
	      s = make_array($1, TYNULL, LLNULL, 0, LOCAL);
              if(s->attr &  DYNAMIC_BIT)
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
              if((s->attr & PROCESSORS_BIT) || (s->attr & TASK_BIT) || (s->attr & HEAP_BIT)) 
                errstr("Inconsistent declaration of identifier  %s", s->ident, 16); 
              else
                s->attr = s->attr | DYNAMIC_BIT;        
	      $$ = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);
	   }
           ;

dvm_inherit: INHERIT in_dcl dummy_array_name_list
    { $$ = get_bfnd(fi,DVM_INHERIT_DIR, SMNULL, $3, LLNULL, LLNULL);}
           ;

dummy_array_name_list: dummy_array_name
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
              | dummy_array_name_list COMMA dummy_array_name
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;

dummy_array_name: name
           {  PTR_SYMB s;
	      s = make_array($1, TYNULL, LLNULL, 0, LOCAL);
              if((s->attr & PROCESSORS_BIT) ||(s->attr & TASK_BIT)  || (s->attr & TEMPLATE_BIT) || (s->attr & ALIGN_BIT) || (s->attr & DISTRIBUTE_BIT)) 
                errstr("Inconsistent declaration of identifier  %s", s->ident, 16); 
              else
                s->attr = s->attr | INHERIT_BIT;        
	      $$ = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);
	   }
           ;

dvm_shadow: SHADOW in_dcl sh_array_name shadow_attr_stuff
           { PTR_LLND q;
             q = set_ll_list($3,LLNULL,EXPR_LIST);
              /* (void)fprintf(stderr,"hpf.gram: shadow\n");*/ 
	     $$ = get_bfnd(fi,DVM_SHADOW_DIR,SMNULL,q,$4,LLNULL);
            }
         /*
          | SHADOW in_dcl shadow_attr_stuff COLON COLON sh_array_name_list
            {$$ = get_bfnd(fi,DVM_SHADOW_DIR,SMNULL,$6,$3,LLNULL);}
          */
          ;
shadow_attr_stuff: LEFTPAR sh_width_list RIGHTPAR
                  { $$ = $2;}
                 ;

sh_width_list: sh_width
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
             | sh_width_list COMMA sh_width
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
             ;

sh_width: expr
          { $$ = $1;}
        | expr COLON expr
          { $$ = make_llnd(fi,DDOT, $1, $3, SMNULL);}
        | COLON LEFTPAR subscript_list RIGHTPAR
          {
            if(parstate!=INEXEC) 
               err("Illegal shadow width specification", 56);  
            $$ = make_llnd(fi,SHADOW_NAMES_OP, $3, LLNULL, SMNULL);
          }
        ;
     
/*         
sh_array_name_list: sh_array_name
	           { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
                  | sh_array_name_list COMMA sh_array_name
	           { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;
*/
sh_array_name: name
           {  PTR_SYMB s;
	      s = make_array($1, TYNULL, LLNULL, 0, LOCAL);
              if(s->attr & SHADOW_BIT)
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
              if((s->attr & PROCESSORS_BIT) ||(s->attr & TASK_BIT)  || (s->attr & TEMPLATE_BIT) || (s->attr & HEAP_BIT)) 
                      errstr( "Inconsistent declaration of identifier %s", s->ident, 16); 
              else
        	      s->attr = s->attr | SHADOW_BIT;  
	      $$ = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);
	   }
           ;  
dvm_processors: HPF_PROCESSORS in_dcl name dims 
	    { PTR_SYMB s;
	      PTR_LLND q, r;
	      if(! explicit_shape) {
              err("Explicit shape specification is required", 50);
		/* $$ = BFNULL;*/
	      }
	      s = make_array($3, TYNULL, $4, ndim, LOCAL);
              if(s->attr &  PROCESSORS_BIT)
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
              if((s->attr & ALIGN_BIT) ||(s->attr & DISTRIBUTE_BIT) ||(s->attr & TEMPLATE_BIT) || (s->attr & DYNAMIC_BIT) ||(s->attr & SHADOW_BIT) || (s->attr & TASK_BIT) || (s->attr & DVM_POINTER_BIT) || (s->attr & INHERIT_BIT))
                errstr("Inconsistent declaration of identifier %s", s->ident, 16);
              else
	        s->attr = s->attr | PROCESSORS_BIT;
              if($4) s->attr = s->attr | DIMENSION_BIT;
	      q = make_llnd(fi,ARRAY_REF, $4, LLNULL, s);
	      s->type->entry.ar_decl.ranges = $4;
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      $$ = get_bfnd(fi,HPF_PROCESSORS_STAT, SMNULL, r, LLNULL, LLNULL);
	    }
	| OMPDVM_NODES in_dcl name dims 
	    { PTR_SYMB s;
	      PTR_LLND q, r;
	      if(! explicit_shape) {
              err("Explicit shape specification is required", 50);
		/* $$ = BFNULL;*/
	      }
	      s = make_array($3, TYNULL, $4, ndim, LOCAL);
              if(s->attr &  PROCESSORS_BIT)
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
              if((s->attr & ALIGN_BIT) ||(s->attr & DISTRIBUTE_BIT) ||(s->attr & TEMPLATE_BIT) || (s->attr & DYNAMIC_BIT) ||(s->attr & SHADOW_BIT) || (s->attr & TASK_BIT) || (s->attr & DVM_POINTER_BIT) || (s->attr & INHERIT_BIT))
                errstr("Inconsistent declaration of identifier %s", s->ident, 16);
              else
	        s->attr = s->attr | PROCESSORS_BIT;
              if($4) s->attr = s->attr | DIMENSION_BIT;
	      q = make_llnd(fi,ARRAY_REF, $4, LLNULL, s);
	      s->type->entry.ar_decl.ranges = $4;
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      $$ = get_bfnd(fi,HPF_PROCESSORS_STAT, SMNULL, r, LLNULL, LLNULL);
	    }
	| dvm_processors COMMA name dims 
           {  PTR_SYMB s;
	      PTR_LLND q, r;
	      if(! explicit_shape) {
		err("Explicit shape specification is required", 50);
		/*$$ = BFNULL;*/
	      }
	      s = make_array($3, TYNULL, $4, ndim, LOCAL);
              if(s->attr &  PROCESSORS_BIT)
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
              if((s->attr & ALIGN_BIT) ||(s->attr & DISTRIBUTE_BIT) ||(s->attr & TEMPLATE_BIT) || (s->attr & DYNAMIC_BIT) ||(s->attr & SHADOW_BIT) || (s->attr & TASK_BIT) || (s->attr &  DVM_POINTER_BIT) || (s->attr & INHERIT_BIT) )
                errstr("Inconsistent declaration of identifier  %s", s->ident, 16);
              else
	        s->attr = s->attr | PROCESSORS_BIT;
              if($4) s->attr = s->attr | DIMENSION_BIT;
	      q = make_llnd(fi,ARRAY_REF, $4, LLNULL, s);
	      s->type->entry.ar_decl.ranges = $4;
	      r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	      add_to_lowLevelList(r, $1->entry.Template.ll_ptr1);
	}
	;

dvm_indirect_group: INDIRECT_GROUP in_dcl indirect_group_name
                {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, $3);
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	           $$ = get_bfnd(fi,DVM_INDIRECT_GROUP_DIR, SMNULL, r, LLNULL, LLNULL);
                }
                  | dvm_indirect_group COMMA  indirect_group_name
                {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, $3);
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
                   add_to_lowLevelList(r, $1->entry.Template.ll_ptr1);
	           ;
                }
                  ;

indirect_group_name: name
          {$$ = make_local_entity($1, REF_GROUP_NAME,global_default,LOCAL);
          if($$->attr &  INDIRECT_BIT)
                errstr( "Multiple declaration of identifier  %s ", $$->ident, 73);
           $$->attr = $$->attr | INDIRECT_BIT;
          }
                   ;

dvm_remote_group: REMOTE_GROUP in_dcl remote_group_name
                {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, $3);
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	           $$ = get_bfnd(fi,DVM_REMOTE_GROUP_DIR, SMNULL, r, LLNULL, LLNULL);
                }
                  | dvm_remote_group COMMA  remote_group_name
                {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, $3);
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
                   add_to_lowLevelList(r, $1->entry.Template.ll_ptr1);
                }
                  ;

remote_group_name: name
          {$$ = make_local_entity($1, REF_GROUP_NAME,global_default,LOCAL);
           if($$->attr &  INDIRECT_BIT)
                errstr( "Inconsistent declaration of identifier  %s ", $$->ident, 16);
          }
          ;

dvm_reduction_group: REDUCTION_GROUP in_dcl reduction_group_name
                {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, $3);
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	           $$ = get_bfnd(fi,DVM_REDUCTION_GROUP_DIR, SMNULL, r, LLNULL, LLNULL);
                }
                  | dvm_reduction_group COMMA  reduction_group_name
                {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, $3);
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
                   add_to_lowLevelList(r, $1->entry.Template.ll_ptr1);
	           ;
                }
                  ;

reduction_group_name:  name
       {$$ = make_local_entity($1, REDUCTION_GROUP_NAME,global_default,LOCAL);}
            ;       

dvm_consistent_group: CONSISTENT_GROUP in_dcl consistent_group_name
                {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, $3);
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	           $$ = get_bfnd(fi,DVM_CONSISTENT_GROUP_DIR, SMNULL, r, LLNULL, LLNULL);
                }
                  | dvm_consistent_group COMMA  consistent_group_name
                {  PTR_LLND q,r;
                   q = make_llnd(fi,VAR_REF, LLNULL, LLNULL, $3);
                   r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
                   add_to_lowLevelList(r, $1->entry.Template.ll_ptr1);	           
                }
                  ;

consistent_group_name:  name
       {$$ = make_local_entity($1, CONSISTENT_GROUP_NAME,global_default,LOCAL);}
            ;       


/*opt_new: 
*	    { $$ = LLNULL; opt_kwd_ = NO;}
*	|    NEW 
*	    { $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
*             $$->entry.string_val = (char *) "new";
*             $$->type = global_string;
*           }
*       ;
*/
opt_onto:   ONTO name dims 
            { PTR_SYMB s;
            if(parstate == INEXEC){
              if (!(s = $2->id_attr))
              {
	         s = make_array($2, TYNULL, LLNULL, 0, LOCAL);
	     	 s->decl = SOFT;
	      } 
            } else
              s = make_array($2, TYNULL, LLNULL, 0, LOCAL);

              $$ = make_llnd(fi,ARRAY_REF, $3, LLNULL, s);
            }               
	|  
            { $$ = LLNULL; opt_kwd_ = NO;}
	    
	;


dvm_distribute: DISTRIBUTE in_dcl dist_name  opt_dist_format_clause opt_onto 
           { PTR_LLND q;
             if(!$4)
               err("Distribution format list is omitted", 51);
            /* if($6)
               err("NEW_VALUE specification in DISTRIBUTE directive");*/
             q = set_ll_list($3,LLNULL,EXPR_LIST);
	     $$ = get_bfnd(fi,DVM_DISTRIBUTE_DIR,SMNULL,q,$4,$5);
            }
      /*        | DISTRIBUTE in_dcl opt_aster opt_key_word opt_dist_format_clause opt_onto COLON COLON dist_name_list
	    { if(!$3 && !$5)
                err(" dist-format-clause is omitted");
              $$ = get_bfnd(fi,DVM_DISTRIBUTE_DIR,$6,$9,$5,$3); }
      */
	      ;

dvm_redistribute: REDISTRIBUTE end_spec  dist_name dist_format_clause opt_key_word opt_onto  
            { PTR_LLND q;
                /*  if(!$4)
                  {err("Distribution format is omitted", 51); errcnt--;}
                 */
              q = set_ll_list($3,LLNULL,EXPR_LIST);
                 /* r = LLNULL;
                   if($6){
                     r = set_ll_list($6,LLNULL,EXPR_LIST);
                     if($7) r = set_ll_list(r,$7,EXPR_LIST);
                   } else
                     if($7) r = set_ll_list(r,$7,EXPR_LIST);
                 */
	      $$ = get_bfnd(fi,DVM_REDISTRIBUTE_DIR,SMNULL,q,$4,$6);}

                | REDISTRIBUTE end_spec  dist_format_clause  opt_key_word opt_onto  COLON COLON dist_name_list
             {
                 /* r = LLNULL;
                    if($5){
                      r = set_ll_list($5,LLNULL,EXPR_LIST);
                      if($6) r = set_ll_list(r,$6,EXPR_LIST);
                    } else
                      if($6) r = set_ll_list(r,$6,EXPR_LIST);
                  */
	      $$ = get_bfnd(fi,DVM_REDISTRIBUTE_DIR,SMNULL,$8 ,$3,$5 );
             }
/*              | REDISTRIBUTE_NEW end_spec dist_name dist_format_clause opt_key_word opt_onto
            { PTR_LLND q,p;
              q = set_ll_list($3,LLNULL,EXPR_LIST);
	      p = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              p->entry.string_val = (char *) "new";
              p->type = global_string;
	      $$ = get_bfnd(fi,DVM_REDISTRIBUTE_DIR,$6,q,$4,p);
            }
               | REDISTRIBUTE_NEW end_spec dist_format_clause  opt_key_word opt_onto COLON COLON dist_name_list
	    { PTR_LLND p;
	      p = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              p->entry.string_val = (char *) "new";
              p->type = global_string;
              $$ = get_bfnd(fi,DVM_REDISTRIBUTE_DIR,$5,$8,$3,p); }
*/    
            ;

dist_name_list: distributee
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
              | dist_name_list COMMA distributee
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;

distributee: dist_name
             {$$ = $1;}
           | pointer_ar_elem
             {$$ = $1;}
           ;

dist_name: name 
       {  PTR_SYMB s;
 
          if(parstate == INEXEC){
            if (!(s = $1->id_attr))
              {
	         s = make_array($1, TYNULL, LLNULL, 0, LOCAL);
	     	 s->decl = SOFT;
	      } 
            if(s->attr & PROCESSORS_BIT)
              errstr( "Illegal use of PROCESSORS name %s ", s->ident, 53);
            if(s->attr & TASK_BIT)
              errstr( "Illegal use of task array name %s ", s->ident, 71);

          } else {
            s = make_array($1, TYNULL, LLNULL, 0, LOCAL);
            if(s->attr & DISTRIBUTE_BIT)
              errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
            else if( (s->attr & PROCESSORS_BIT) || (s->attr & TASK_BIT) || (s->attr & INHERIT_BIT))
              errstr("Inconsistent declaration of identifier  %s",s->ident, 16);
            else
              s->attr = s->attr | DISTRIBUTE_BIT;
          } 
         if(s->attr & ALIGN_BIT)
               errstr("A distributee may not have the ALIGN attribute:%s",s->ident, 54);
          $$ = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);               	  
	}
        ;

pointer_ar_elem: name LEFTPAR subscript_list RIGHTPAR
       {  PTR_SYMB s;
          s = make_array($1, TYNULL, LLNULL, 0, LOCAL);
        
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
	  $$ = make_llnd(fi,ARRAY_REF, $3, LLNULL, s); 
          }
        
	}
        ;


processors_name: name 
       {  PTR_SYMB s;
          if((s=$1->id_attr) == SMNULL)
            s = make_array($1, TYNULL, LLNULL, 0, LOCAL);
          if((parstate == INEXEC) && !(s->attr & PROCESSORS_BIT))
               errstr( "'%s' is not processor array ", s->ident, 67);
	  $$ = s;
	}
        ;
/*
opt_aster:
          {$$ = LLNULL;} 
         | ASTER
          { 
            $$ = make_llnd(fi,KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
            $$->entry.string_val = (char *) "*";
            $$->type = global_string;
          }
          ; 
*/
opt_dist_format_clause: 
                       { $$ = LLNULL;  }
                      | dist_format_clause opt_key_word 
                        { $$ = $1;}
                      ; 
 
dist_format_clause: LEFTPAR dist_format_list RIGHTPAR
                      { $$ = $2;}
                  ;
/*opt_dist_format_clause: dist_format_clause
                      { (void)fprintf(stderr,"hpf.gram:opt_dist_format_clause\n");
                         $$ = $1;}
                      | ASTER dist_format_clause
                      { PTR_LLND q;
                         q = make_llnd(fi,KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
                         q->entry.string_val = (char *) "*";
                         q->type = global_string;
                      (void)fprintf(stderr,"hpf.gram: * opt_dist_format_clause\n");
                       $$ = make_llnd(fi,ARRAY_OP, q, $2, SMNULL);} 
                      | ASTER 
           {         
             $$ = make_llnd(fi,KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
             $$->entry.string_val = (char *) "*";
             $$->type = global_string;
           } 
                  ; 
*/
dist_format_list: opt_key_word dist_format
	    { $$ = set_ll_list($2,LLNULL,EXPR_LIST); }
                | dist_format_list COMMA opt_key_word dist_format
	    { $$ = set_ll_list($1,$4,EXPR_LIST); }	    
	        ;
opt_key_word:
            { opt_kwd_ = YES; }
            ;
/*
no_opt_key_word:
            { opt_kwd_ = NO; }
	     ;	     
*/

dist_format:   BLOCK
        {  
               $$ = make_llnd(fi,BLOCK_OP, LLNULL, LLNULL, SMNULL);
        }
           |   BLOCK LEFTPAR in_ioctl shadow_width  RIGHTPAR
        {  err("Distribution format BLOCK(n) is not permitted in FDVM", 55);
          $$ = make_llnd(fi,BLOCK_OP, $4, LLNULL, SMNULL);
          endioctl();
        }
           |   GEN_BLOCK LEFTPAR array_name RIGHTPAR
        { $$ = make_llnd(fi,BLOCK_OP, LLNULL, LLNULL, $3); }  
           |   WGT_BLOCK LEFTPAR array_name COMMA expr RIGHTPAR
        { $$ = make_llnd(fi,BLOCK_OP,  $5,  LLNULL,  $3); }   
           |   MULT_BLOCK LEFTPAR  expr RIGHTPAR
        { $$ = make_llnd(fi,BLOCK_OP,  LLNULL, $3,  SMNULL); } 
           |  ASTER
        { 
          $$ = make_llnd(fi,KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
          $$->entry.string_val = (char *) "*";
          $$->type = global_string;
        }
           |  INDIRECT LEFTPAR array_name RIGHTPAR 
        { $$ = make_llnd(fi,INDIRECT_OP, LLNULL, LLNULL, $3); }     
           |  DERIVED LEFTPAR derived_spec RIGHTPAR 
        { $$ = make_llnd(fi,INDIRECT_OP, $3, LLNULL, SMNULL); }     
           ;

array_name: name
           {  PTR_SYMB s;
	      s = make_array($1, TYNULL, LLNULL, 0, LOCAL);
              if((s->attr & PROCESSORS_BIT) ||(s->attr & TASK_BIT)  || (s->attr & TEMPLATE_BIT)) 
                errstr("Inconsistent declaration of identifier  %s", s->ident, 16); 
       
	      $$ = s;
	   }
           ;

derived_spec: LEFTPAR derived_elem_list  RIGHTPAR needkeyword WITH target_spec 
            { $$ = make_llnd(fi,DERIVED_OP, $2, $6, SMNULL); }
            ;

derived_elem_list: derived_elem 
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
             | derived_elem_list COMMA derived_elem 
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    

             ;

derived_elem: expr
              { $$ = $1;}
            | expr COLON expr
              { $$ = make_llnd(fi,DDOT, $1, $3, SMNULL);} 
            ;

target_spec: derived_target 
            { 
              $$ = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, $1);
	    }
            | derived_target LEFTPAR derived_subscript_list RIGHTPAR
            { 
              $$ = make_llnd(fi,ARRAY_REF, $3, LLNULL, $1);
	    }
            ;

derived_target:  name 
            { 
              if (!($$ = $1->id_attr))
              {
	         $$ = make_array($1, TYNULL, LLNULL,0,LOCAL);
	         $$->decl = SOFT;
	      } 
            }
            ;

derived_subscript_list: derived_subscript
	                { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
                      | derived_subscript_list COMMA derived_subscript
	                { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
                      ;

derived_subscript:  expr
                   {  $$ = $1;}
                 |  aster_expr
                   {  $$ = $1;}
                 |  AT dummy_ident opt_plus_shadow
                   {
                      $2->entry.Template.ll_ptr1 = $3; 
                      $$ = $2;   
                   }     
                 ;

dummy_ident: name
          { PTR_SYMB s;
            s = make_scalar($1,TYNULL,LOCAL);
	    $$ = make_llnd(fi,DUMMY_REF, LLNULL, LLNULL, s);
            /*$$->type = global_int;*/
          }
          ; 
/*
dummy_ident: IDENTIFIER
           {
                    PTR_HASH hash_entry;
                    hash_entry = just_look_up_sym_in_scope(cur_scope(),yytext);
                    if
           }
           ;
*/

opt_plus_shadow: 
               {  $$ = LLNULL; }
               | plus_shadow
               {  $$ = $1; }
               ; 

plus_shadow: PLUS shadow_id
           {  $$ = set_ll_list($2,LLNULL,EXPR_LIST); }
           | opt_plus_shadow PLUS shadow_id
           {  $$ = set_ll_list($1,$3,EXPR_LIST); }
           ;
 
shadow_id:  expr
            {  if($1->type->variant != T_STRING)
                 errstr( "Illegal type of shadow_name", 627);
               $$ = $1; 
            }
         ;

shadow_width: nameeq expr
        { char *q;
          nioctl = 1;
          q = $1->entry.string_val;
          if((!strcmp(q,"shadow")) && ($2->variant == INT_VAL))                          $$ = make_llnd(fi,SPEC_PAIR, $1, $2, SMNULL);
          else
          {  err("Illegal shadow width specification", 56);
             $$ = LLNULL;
          }
        }
           | nameeq expr COMMA nameeq expr 
        { char *ql, *qh;
          PTR_LLND p1, p2;
          nioctl = 2;
          ql = $1->entry.string_val;
          qh = $4->entry.string_val;
          if((!strcmp(ql,"low_shadow")) && ($2->variant == INT_VAL) && (!strcmp(qh,"high_shadow")) && ($5->variant == INT_VAL)) 
              {
                 p1 = make_llnd(fi,SPEC_PAIR, $1, $2, SMNULL);
                 p2 = make_llnd(fi,SPEC_PAIR, $4, $5, SMNULL);
                 $$ = make_llnd(fi,CONS, p1, p2, SMNULL);
              } 
          else
          {  err("Illegal shadow width specification", 56);
             $$ = LLNULL;
          }
        }
          ;

/*
opt_new_value_spec:
                     {$$ = LLNULL;}
                  |  COMMA needkeyword NEW_VALUE
                     {$$ = make_llnd(fi,NEW_VALUE_OP, LLNULL, LLNULL, SMNULL);}
                  |  COMMA needkeyword NEW_VALUE LEFTPAR  array_ident_list RIGHTPAR
                     {$$ = make_llnd(fi,NEW_VALUE_OP, $5, LLNULL, SMNULL);}
                  ;
*/

dvm_align:  ALIGN in_dcl alignee align_directive_stuff
	    { PTR_LLND q;
              q = set_ll_list($3,LLNULL,EXPR_LIST);
              $$ = $4;
              $$->entry.Template.ll_ptr1 = q;
            }
           /*
            *|  ALIGN in_dcl  align_directive_stuff COLON COLON alignee_list
	    * { 
            *   $$ = $3;
            *   $$->entry.Template.ll_ptr1 = $6;
            * }
            */
	;

dvm_realign:  REALIGN end_spec realignee align_directive_stuff 
	    { PTR_LLND q;
              q = set_ll_list($3,LLNULL,EXPR_LIST);
              $$ = $4;
              $$->variant = DVM_REALIGN_DIR; 
              $$->entry.Template.ll_ptr1 = q;
            }
           |  REALIGN end_spec  align_directive_stuff COLON COLON realignee_list
	    {
              $$ = $3;
              $$->variant = DVM_REALIGN_DIR; 
              $$->entry.Template.ll_ptr1 = $6;
            }
        /*
           |  REALIGN_WITH end_spec  align_base COLON COLON realignee_list
	    { $$ = get_bfnd(fi,DVM_REALIGN_DIR,SMNULL,$6,LLNULL,$3);}
         */
	   ;
/*
alignee_list: alignee
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
              | alignee_list COMMA alignee
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;
*/
realignee_list: realignee
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
              | realignee_list COMMA realignee
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;

alignee: name
       {  PTR_SYMB s;
          s = make_array($1, TYNULL, LLNULL, 0, LOCAL);
          if((s->attr & ALIGN_BIT)) 
                errstr( "Multiple declaration of identifier  %s ", s->ident, 73);
          if((s->attr & PROCESSORS_BIT) || (s->attr & TASK_BIT) || (s->attr & INHERIT_BIT)) 
                errstr( "Inconsistent declaration of identifier  %s", s->ident, 16); 
          else  if(s->attr & DISTRIBUTE_BIT)
               errstr( "An alignee may not have the DISTRIBUTE attribute:'%s'", s->ident,57);             else
                s->attr = s->attr | ALIGN_BIT;     
	  $$ = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);
	}
        ;
  
realignee: ident
       {PTR_SYMB s;
        s = $1->entry.Template.symbol;
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
	  $$ = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);
	}
        ;

align_directive_stuff: LEFTPAR dim_ident_list RIGHTPAR needkeyword WITH align_base 
           { /* PTR_LLND r;
              if($7) {
                r = set_ll_list($6,LLNULL,EXPR_LIST);
                r = set_ll_list(r,$7,EXPR_LIST);
              }
              else
                r = $6;
              */
            $$ = get_bfnd(fi,DVM_ALIGN_DIR,SMNULL,LLNULL,$2,$6);
           }  
                     ;

align_base:  align_base_name LEFTPAR align_subscript_list RIGHTPAR
	  {
           $$ = make_llnd(fi,ARRAY_REF, $3, LLNULL, $1);        
          } 
/*
          |  align_base_name
          {PTR_LLND q;
           q = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, $2);
           if($1)
               $$ = make_llnd(fi,ARRAY_OP, $1, q, SMNULL);
           else
               $$ = q;
          } 
*/
          ;

align_subscript_list: align_subscript
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
              | align_subscript_list COMMA align_subscript
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	
                    ;
align_subscript:  expr
                 { $$ = $1;}
               | ASTER
                 {
                  $$ = make_llnd(fi,KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
                  $$->entry.string_val = (char *) "*";
                  $$->type = global_string;
                 }
               | triplet
                 { $$ = $1;}
               ;
 
align_base_name: name
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
          if (!($$ = $1->id_attr))
          {
	       $$ = make_array($1, TYNULL, LLNULL,0,LOCAL);
	       $$->decl = SOFT;
	  } 
          $$->attr = $$->attr | ALIGN_BASE_BIT;
          if($$->attr & PROCESSORS_BIT)
               errstr( "Illegal use of PROCESSORS name %s ", $$->ident, 53);
          else  if($$->attr & TASK_BIT)
               errstr( "Illegal use of task array name %s ", $$->ident, 71);
          else
          if((parstate == INEXEC) /* for  REALIGN directive */
             &&   !($$->attr & DIMENSION_BIT) && !($$->attr & DVM_POINTER_BIT))
            errstr("The align-target %s isn't declared as array", $$->ident, 61); 
         }  
              ;

dim_ident_list: dim_ident
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
              | dim_ident_list COMMA dim_ident
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;

dim_ident: name
          { PTR_SYMB s;
            s = make_scalar($1,TYNULL,LOCAL);
            if(s->type->variant != T_INT || s->attr & PARAMETER_BIT)             
              errstr("The align-dummy %s isn't a scalar integer variable", s->ident, 62); 
	   $$ = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
           $$->type = global_int;
         }  
           |  ASTER
        {  
          $$ = make_llnd(fi,KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
          $$->entry.string_val = (char *) "*";
          $$->type = global_string;
        }
           |  COLON
        {   $$ = make_llnd(fi,DDOT, LLNULL, LLNULL, SMNULL); }   
        ;
dvm_combined_dir: dvm_attribute_list COLON COLON  in_dcl name dims
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
                     if($6)          /*dimension information after the object name*/
                     { p = $6; numdim = ndim;} /*overrides the DIMENSION attribute */
	             s = make_array($5, TYNULL, p, numdim, LOCAL);

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
                     if($6) s->attr = s->attr | DIMENSION_BIT;
                     if((s->attr & DISTRIBUTE_BIT) && (s->attr & ALIGN_BIT))
                       errstr("%s has the DISTRIBUTE and ALIGN attribute",s->ident, 64);
	             q = make_llnd(fi,ARRAY_REF, $6, LLNULL, s);
	             if(p) s->type->entry.ar_decl.ranges = p;
	             r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	             $$ = get_bfnd(fi,DVM_VAR_DECL, SMNULL, r, LLNULL,$1);
	            }
                | dvm_combined_dir COMMA name dims
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
                     if($4)                   /*dimension information after the object name*/
                     { p = $4; numdim = ndim;}/*overrides the DIMENSION attribute */
	             s = make_array($3, TYNULL, p, numdim, LOCAL);

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
                     if($4) s->attr = s->attr | DIMENSION_BIT;
                     if((s->attr & DISTRIBUTE_BIT) && (s->attr & ALIGN_BIT))
                           errstr("%s has the DISTRIBUTE and ALIGN attribute",s->ident, 64);
	             q = make_llnd(fi,ARRAY_REF, $4, LLNULL, s);
	             if(p) s->type->entry.ar_decl.ranges = p;
	             r = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);
	             add_to_lowLevelList(r, $1->entry.Template.ll_ptr1);
	            }
                ;
         
/*dvm_attribute_list: { type_options = 0; attr_ndim = 0; attr_dims = LLNULL;} dvm_attribute
	    { $$ = set_ll_list($2,LLNULL,EXPR_LIST); }
              | dvm_attribute_list COMMA needkeyword dvm_attribute
	    { $$ = set_ll_list($1,$4,EXPR_LIST); }	
                 ;
*/
dvm_attribute_list:  dvm_attribute
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); type_options = type_opt; }
              | dvm_attribute_list COMMA needkeyword dvm_attribute
	    { $$ = set_ll_list($1,$4,EXPR_LIST); type_options = type_options | type_opt;}
                 ;
dvm_attribute: HPF_TEMPLATE
               { type_opt = TEMPLATE_BIT;
               $$ = make_llnd(fi,TEMPLATE_OP,LLNULL,LLNULL,SMNULL);
               } 
             | HPF_PROCESSORS
               { type_opt = PROCESSORS_BIT;
                $$ = make_llnd(fi,PROCESSORS_OP,LLNULL,LLNULL,SMNULL);
               } 
             | OMPDVM_NODES
               { type_opt = PROCESSORS_BIT;
                $$ = make_llnd(fi,PROCESSORS_OP,LLNULL,LLNULL,SMNULL);
               } 
             | DYNAMIC
               { type_opt = DYNAMIC_BIT;
                $$ = make_llnd(fi,DYNAMIC_OP,LLNULL,LLNULL,SMNULL);
               } 
            /* | DIMENSION dims
               {
                if(! explicit_shape) {
                  err("Explicit shape specification is required", 50);
                }
                if(! $2) {
                  err("No shape specification", 65);
	        }
                type_opt = DIMENSION_BIT;
                attr_ndim = ndim; attr_dims = $2;
                $$ = make_llnd(fi,DIMENSION_OP,$2,LLNULL,SMNULL);
	       }*/

            | DIMENSION LEFTPAR dimlist RIGHTPAR
               {
                if(! explicit_shape) {
                  err("Explicit shape specification is required", 50);
                }
                if(! $3) {
                  err("No shape specification", 65);
	        }
                type_opt = DIMENSION_BIT;
                attr_ndim = ndim; attr_dims = $3;
                $$ = make_llnd(fi,DIMENSION_OP,$3,LLNULL,SMNULL);
	       }
              | SHADOW shadow_attr_stuff
                { type_opt = SHADOW_BIT;
                  $$ = make_llnd(fi,SHADOW_OP,$2,LLNULL,SMNULL);
                 } 
              | ALIGN LEFTPAR dim_ident_list RIGHTPAR needkeyword WITH align_base
                 { type_opt = ALIGN_BIT;
                  $$ = make_llnd(fi,ALIGN_OP,$3,$7,SMNULL);
                 } 
              | ALIGN 
                { type_opt = ALIGN_BIT;
                  $$ = make_llnd(fi,ALIGN_OP,LLNULL,SMNULL,SMNULL);
                }
/*
              | ALIGN_WITH align_base
                { type_opt = ALIGN_BIT;
                  $$ = make_llnd(fi,ALIGN_OP,LLNULL,$2,SMNULL);
                }
*/
              | DISTRIBUTE  dist_format_clause opt_key_word  opt_onto 
	        { 
                 type_opt = DISTRIBUTE_BIT;
                 $$ = make_llnd(fi,DISTRIBUTE_OP,$2,$4,SMNULL);
                } 
              | DISTRIBUTE  
	        { 
                 type_opt = DISTRIBUTE_BIT;
                 $$ = make_llnd(fi,DISTRIBUTE_OP,LLNULL,LLNULL,SMNULL);
                } 
              | COMMON
                {
                 type_opt = COMMON_BIT;
                 $$ = make_llnd(fi,COMMON_OP, LLNULL, LLNULL, SMNULL);
                }
               ; 

dvm_pointer: type COMMA needkeyword DVM_POINTER in_dcl LEFTPAR dimension_list RIGHTPAR COLON COLON pointer_var_list
           { 
	      PTR_LLND  l;
	      l = make_llnd(fi, TYPE_OP, LLNULL, LLNULL, SMNULL);
	      l->type = $1;
	      $$ = get_bfnd(fi,DVM_POINTER_DIR, SMNULL, $11,$7, l);
	    }
           ;

dimension_list: {ndim = 0;} COLON
	    { PTR_LLND  q;
             if(ndim == maxdim)
		err("Too many dimensions", 43);
	      else if(ndim < maxdim)
		q = make_llnd(fi,DDOT,LLNULL,LLNULL,SMNULL);
	      ++ndim;
              $$ = set_ll_list(q, LLNULL, EXPR_LIST);
	       /*$$ = make_llnd(fi,EXPR_LIST, q, LLNULL, SMNULL);*/
	       /*$$->type = global_default;*/
	    }
	| dimension_list COMMA COLON
	    { PTR_LLND  q;
             if(ndim == maxdim)
		err("Too many dimensions", 43);
	      else if(ndim < maxdim)
		q = make_llnd(fi,DDOT,LLNULL,LLNULL,SMNULL);
	      ++ndim;
              $$ = set_ll_list($1, q, EXPR_LIST);
            }
	;

pointer_var_list: pointer_var
	          { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
                |  pointer_var_list COMMA  pointer_var
	          { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	        ;

pointer_var: name
           {PTR_SYMB s;
           /* s = make_scalar($1,TYNULL,LOCAL);*/
            s = make_array($1,TYNULL,LLNULL,0,LOCAL);
            s->attr = s->attr | DVM_POINTER_BIT;
            if((s->attr & PROCESSORS_BIT) || (s->attr & TASK_BIT) || (s->attr & INHERIT_BIT))
               errstr( "Inconsistent declaration of identifier %s", s->ident, 16);     
            $$ = make_llnd(fi,VAR_REF,LLNULL,LLNULL,s);
            }
            ;

dvm_heap: HEAP in_dcl  heap_array_name_list
           { $$ = get_bfnd(fi,DVM_HEAP_DIR, SMNULL, $3, LLNULL, LLNULL);}
           ;

heap_array_name_list: heap_array_name
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
              | heap_array_name_list COMMA heap_array_name
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;

heap_array_name: name
           {  PTR_SYMB s;
	      s = make_array($1, TYNULL, LLNULL, 0, LOCAL);
              s->attr = s->attr | HEAP_BIT;
              if((s->attr & PROCESSORS_BIT) ||(s->attr & TASK_BIT)  || (s->attr & TEMPLATE_BIT) || (s->attr & ALIGN_BIT) || (s->attr & DISTRIBUTE_BIT) || (s->attr & INHERIT_BIT) || (s->attr & DYNAMIC_BIT) || (s->attr & SHADOW_BIT) || (s->attr & DVM_POINTER_BIT)) 
                errstr("Inconsistent declaration of identifier  %s", s->ident, 16); 
      
	      $$ = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);
	   }
           ;

dvm_consistent: CONSISTENT in_dcl  consistent_array_name_list
           { $$ = get_bfnd(fi,DVM_CONSISTENT_DIR, SMNULL, $3, LLNULL, LLNULL);}
           ;

consistent_array_name_list: consistent_array_name
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
              | consistent_array_name_list COMMA consistent_array_name
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;

consistent_array_name: name
           {  PTR_SYMB s;
	      s = make_array($1, TYNULL, LLNULL, 0, LOCAL);
              s->attr = s->attr | CONSISTENT_BIT;
              if((s->attr & PROCESSORS_BIT) ||(s->attr & TASK_BIT)  || (s->attr & TEMPLATE_BIT) || (s->attr & ALIGN_BIT) || (s->attr & DISTRIBUTE_BIT) || (s->attr & INHERIT_BIT) || (s->attr & DYNAMIC_BIT) || (s->attr & SHADOW_BIT) || (s->attr & DVM_POINTER_BIT)) 
                errstr("Inconsistent declaration of identifier  %s", s->ident, 16); 
      
	      $$ = make_llnd(fi,ARRAY_REF, LLNULL, LLNULL, s);
	   }
           ;


dvm_asyncid: ASYNCID in_dcl  async_id_list
            { $$ = get_bfnd(fi,DVM_ASYNCID_DIR, SMNULL, $3, LLNULL, LLNULL);}
	   | ASYNCID in_dcl COMMA needkeyword COMMON COLON COLON async_id_list 
            { PTR_LLND p;
              p = make_llnd(fi,COMM_LIST, LLNULL, LLNULL, SMNULL);              
              $$ = get_bfnd(fi,DVM_ASYNCID_DIR, SMNULL, $8, p, LLNULL);
            }
           ;

async_id_list: async_id
	       { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
             | async_id_list COMMA async_id
	       { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	     ;

async_id: name dims
           {  PTR_SYMB s;
              if($2){
                  s = make_array($1, global_default, $2, ndim, LOCAL);
		  s->variant = ASYNC_ID;
                  s->attr = s->attr | DIMENSION_BIT;
                  s->type->entry.ar_decl.ranges = $2;
                  $$ = make_llnd(fi,ARRAY_REF, $2, LLNULL, s);
              } else {
              s = make_local_entity($1, ASYNC_ID, global_default, LOCAL);
	      $$ = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
              }
	   }
           ;


dvm_new_value: NEW_VALUE  end_spec
               { $$ = get_bfnd(fi,DVM_NEW_VALUE_DIR,SMNULL, LLNULL, LLNULL,LLNULL);} 
          /* | NEW_VALUE  end_spec array_ident_list
               { $$ = get_bfnd(fi,DVM_NEW_VALUE_DIR,SMNULL, $3, LLNULL,LLNULL);}
           */
             ;


dvm_parallel_on: PARALLEL end_spec LEFTPAR ident_list RIGHTPAR opt_on opt_spec 
 

                 {  if($6 &&  $6->entry.Template.symbol->attr & TASK_BIT)
                        $$ = get_bfnd(fi,DVM_PARALLEL_TASK_DIR,SMNULL,$6,$7,$4);
                    else
                        $$ = get_bfnd(fi,DVM_PARALLEL_ON_DIR,SMNULL,$6,$7,$4);
                 }
               ;


ident_list: ident
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
             | ident_list COMMA ident
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;

opt_on:   opt_key_word ON distribute_cycles 
          { $$ = $3;}
                       
	| opt_key_word 
            { $$ = LLNULL; opt_kwd_ = NO;}
	    
	;

distribute_cycles: ident LEFTPAR par_subscript_list RIGHTPAR
        {
          if($1->type->variant != T_ARRAY) 
             errstr("'%s' isn't array", $1->entry.Template.symbol->ident, 66);
          $1->entry.Template.ll_ptr1 = $3;
          $$ = $1;
          $$->type = $1->type->entry.ar_decl.base_type;
        }
        ;

par_subscript_list: par_subscript
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
              | par_subscript_list COMMA par_subscript
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;

par_subscript: expr
            { $$ = $1;}
             | ASTER
            {
             $$ = make_llnd(fi,KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
             $$->entry.string_val = (char *) "*";
             $$->type = global_string;
            }
             ;

opt_spec:
          {  $$ = LLNULL;}
        | spec_list
          { $$ = $1;}
        ;

spec_list:     par_spec
	       { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
             | spec_list par_spec
	       { $$ = set_ll_list($1,$2,EXPR_LIST); }	
             ;
           

par_spec: new_spec
        | reduction_spec
        | shadow_spec
        | remote_access_spec
        | indirect_access_spec
        | across_spec
        | stage_spec
        | consistent_spec
        | private_spec     /*ACC*/
        | cuda_block_spec  /*ACC*/
        | tie_spec         /*ACC*/
        ;

remote_access_spec:  COMMA needkeyword REMOTE_ACCESS_SPEC LEFTPAR group_name COLON remote_data_list RIGHTPAR
                        { if($5->attr & INDIRECT_BIT)
                            errstr("'%s' is not remote group name", $5->ident, 68);
                          $$ = make_llnd(fi,REMOTE_ACCESS_OP,$7,LLNULL,$5);
                        }
                  |  COMMA needkeyword REMOTE_ACCESS_SPEC LEFTPAR remote_data_list RIGHTPAR
                        { $$ = make_llnd(fi,REMOTE_ACCESS_OP,$5,LLNULL,SMNULL);}
                      ;

consistent_spec:  COMMA needkeyword CONSISTENT_SPEC LEFTPAR consistent_group COLON remote_data_list RIGHTPAR
                        {
                          $$ = make_llnd(fi,CONSISTENT_OP,$7,LLNULL,$5);
                        }
                  |  COMMA needkeyword CONSISTENT_SPEC LEFTPAR remote_data_list RIGHTPAR
                        { $$ = make_llnd(fi,CONSISTENT_OP,$5,LLNULL,SMNULL);}
                      ;

consistent_group: name
          {  
            if(($$=$1->id_attr) == SMNULL){
                errstr("'%s' is not declared as group", $1->ident, 74);
                $$ = make_local_entity($1,CONSISTENT_GROUP_NAME,global_default,LOCAL);
            } else {
                if($$->variant != CONSISTENT_GROUP_NAME)
                   errstr("'%s' is not declared as group", $1->ident, 74);
            }
          }
          ;


new_spec: COMMA needkeyword NEW LEFTPAR variable_list RIGHTPAR
              {$$ = make_llnd(fi,NEW_SPEC_OP,$5,LLNULL,SMNULL);}
/*	| COMMA needkeyword PRIVATE LEFTPAR variable_list RIGHTPAR
	      {$$ = make_llnd(fi,NEW_SPEC_OP,$5,LLNULL,SMNULL);} */  /*OMP*/
        | COMMA needkeyword OMPDVM_FIRSTPRIVATE LEFTPAR variable_list RIGHTPAR
              {$$ = make_llnd(fi,NEW_SPEC_OP,$5,LLNULL,SMNULL);}   /*OMP*/
        ;

private_spec: COMMA needkeyword PRIVATE LEFTPAR variable_list RIGHTPAR
               { $$ = make_llnd(fi,ACC_PRIVATE_OP,$5,LLNULL,SMNULL);}  /*ACC*/
            ;

cuda_block_spec: COMMA needkeyword ACC_CUDA_BLOCK LEFTPAR sizelist RIGHTPAR
                { $$ = make_llnd(fi,ACC_CUDA_BLOCK_OP,$5,LLNULL,SMNULL);} /*ACC*/
               ;
sizelist:  expr
           { $$ = set_ll_list($1,LLNULL,EXPR_LIST);}        /*ACC*/
        |  expr COMMA expr
           {$$ = set_ll_list($1,$3,EXPR_LIST);}            /*ACC*/
        |  expr COMMA expr COMMA expr
           {$$ = set_ll_list($1,$3,EXPR_LIST); $$ = set_ll_list($$,$5,EXPR_LIST);} /*ACC*/
        ;

variable_list: lhs
               { $$ = set_ll_list($1,LLNULL,EXPR_LIST);}
             | variable_list COMMA lhs
               { $$ = set_ll_list($1,$3,EXPR_LIST);}
             ;

tie_spec: COMMA needkeyword ACC_TIE LEFTPAR tied_array_list RIGHTPAR
         { $$ = make_llnd(fi,ACC_TIE_OP,$5,LLNULL,SMNULL);} /*ACC*/
        ;

tied_array_list: distribute_cycles
                 { $$ = set_ll_list($1,LLNULL,EXPR_LIST);}
               | tied_array_list COMMA distribute_cycles
                 { $$ = set_ll_list($1,$3,EXPR_LIST);}
               ;

indirect_access_spec: COMMA needkeyword INDIRECT_ACCESS LEFTPAR group_name COLON indirect_list RIGHTPAR 
                    { if(!($5->attr & INDIRECT_BIT))
                         errstr("'%s' is not indirect group name", $5->ident, 313);
                      $$ = make_llnd(fi,INDIRECT_ACCESS_OP,$7,LLNULL,$5);
                    }
                   |  COMMA needkeyword INDIRECT_ACCESS LEFTPAR  indirect_list RIGHTPAR 
                        { $$ = make_llnd(fi,INDIRECT_ACCESS_OP,$5,LLNULL,SMNULL);}
                   ;

stage_spec:  COMMA needkeyword STAGE LEFTPAR expr RIGHTPAR
              {$$ = make_llnd(fi,STAGE_OP,$5,LLNULL,SMNULL);}
          ;

across_spec: COMMA needkeyword ACROSS in_out_across
               {$$ = make_llnd(fi,ACROSS_OP,$4,LLNULL,SMNULL);}
           | COMMA needkeyword ACROSS in_out_across in_out_across
               {$$ = make_llnd(fi,ACROSS_OP,$4,$5,SMNULL);}
           ;

in_out_across:  LEFTPAR opt_keyword_in_out opt_in_out dependent_array_list RIGHTPAR
                {  if($3)
                     $$ = make_llnd(fi,DDOT,$3,$4,SMNULL);
                   else
                     $$ = $4;
                }
             ;

opt_keyword_in_out:
            { opt_in_out = YES; }
            ;

opt_in_out:  IN  COLON
            {
	      $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "in";
              $$->type = global_string;
            }
          |  OUT COLON
            {
	      $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "out";
              $$->type = global_string;
            }
          |
            {  $$ = LLNULL; opt_in_out = NO;}
          ;

dependent_array_list: dependent_array
            { $$ = set_ll_list($1,LLNULL,EXPR_LIST);}
             | dependent_array_list COMMA dependent_array 
            { $$ = set_ll_list($1,$3,EXPR_LIST);}
             ;

dependent_array: array_ident
                 { $$ = $1;}
               | array_ident LEFTPAR dependence_list RIGHTPAR
                  { $$ = $1;
                    $$-> entry.Template.ll_ptr1 = $3;  
                  }
               | array_ident LEFTPAR dependence_list RIGHTPAR LEFTPAR section_spec_list RIGHTPAR
                 { /*  PTR_LLND p;
                       p = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
                       p->entry.string_val = (char *) "corner";
                       p->type = global_string;
                   */
                   $1-> entry.Template.ll_ptr1 = $3;  
                   $$ = make_llnd(fi,ARRAY_OP,$1,$6,SMNULL);
                 }

               ;  
 
dependence_list: dependence
            { $$ = set_ll_list($1,LLNULL,EXPR_LIST);}
             | dependence_list COMMA dependence 
            { $$ = set_ll_list($1,$3,EXPR_LIST);}
             ; 
  
dependence:  expr COLON expr
             { $$ = make_llnd(fi,DDOT, $1, $3, SMNULL);}
          ;      

section_spec_list: section_spec
              { $$ = set_ll_list($1,LLNULL,EXPR_LIST);}
            | section_spec_list COMMA  section_spec
              { $$ = set_ll_list($1,$3,EXPR_LIST);}
            ;

section_spec: ar_section COLON low_section COLON high_section 
             { $$ = make_llnd(fi,DDOT,$1,make_llnd(fi,DDOT,$3,$5,SMNULL),SMNULL); } 
            |  ar_section COLON low_section  
             { $$ = make_llnd(fi,DDOT,$1,make_llnd(fi,DDOT,$3,LLNULL,SMNULL),SMNULL); } 
            | ar_section COLON high_section
             { $$ = make_llnd(fi,DDOT,$1,make_llnd(fi,DDOT,LLNULL,$3,SMNULL),SMNULL); } 
            | ar_section
             { $$ = make_llnd(fi,DDOT,$1,LLNULL,SMNULL); }
            | low_section COLON high_section
             { $$ = make_llnd(fi,DDOT,LLNULL,make_llnd(fi,DDOT,$1,$3,SMNULL),SMNULL); }
            | low_section
             { $$ = make_llnd(fi,DDOT,LLNULL,make_llnd(fi,DDOT,$1,LLNULL,SMNULL),SMNULL); }
            |  high_section
             { $$ = make_llnd(fi,DDOT,LLNULL,make_llnd(fi,DDOT,LLNULL,$1,SMNULL),SMNULL); }
            ;

ar_section: needkeyword SECTION section
            { $$ = $3;}
          ;

low_section: needkeyword LOW section
            { $$ = $3;}
          ;

high_section: needkeyword HIGH section
            { $$ = $3;}
          ;

section:  LEFTPAR funargs RIGHTPAR
          { $$ = $2;}
       ;

reduction_spec: COMMA needkeyword REDUCTION LEFTPAR  opt_key_word_r  reduction  no_opt_key_word_r  COMMA reduction_list RIGHTPAR
                {PTR_LLND q;
                /* q = set_ll_list($9,$6,EXPR_LIST); */
                 q = set_ll_list($6,LLNULL,EXPR_LIST); /*podd 11.10.01*/
                 q = add_to_lowLevelList($9,q);        /*podd 11.10.01*/
                 $$ = make_llnd(fi,REDUCTION_OP,q,LLNULL,SMNULL);
                }
              | COMMA needkeyword REDUCTION LEFTPAR  opt_key_word_r  reduction  no_opt_key_word_r  RIGHTPAR
                {PTR_LLND q;
                 q = set_ll_list($6,LLNULL,EXPR_LIST);
                 $$ = make_llnd(fi,REDUCTION_OP,q,LLNULL,SMNULL);
                }

              | COMMA needkeyword REDUCTION LEFTPAR opt_key_word_r reduction_group no_opt_key_word_r COLON reduction_list RIGHTPAR
                {  $$ = make_llnd(fi,REDUCTION_OP,$9,LLNULL,$6); }            
              ;

opt_key_word_r:
                  { opt_kwd_r = YES; }
               ;  
no_opt_key_word_r:
                  { opt_kwd_r = NO; }
               ;  

reduction_group:  name
                { 
                  if(($$=$1->id_attr) == SMNULL) {
                      errstr("'%s' is not declared as reduction group", $1->ident, 69);
                      $$ = make_local_entity($1,REDUCTION_GROUP_NAME,global_default,LOCAL);
                  } else {
                    if($$->variant != REDUCTION_GROUP_NAME)
                      errstr("'%s' is not declared as reduction group", $1->ident, 69);
                  }
                }
               ;       


reduction_list: needkeyword reduction
               {$$ = set_ll_list($2,LLNULL,EXPR_LIST);}
              | reduction_list COMMA  needkeyword reduction
               {$$ = set_ll_list($1,$4,EXPR_LIST);}
             ;

reduction: reduction_op LEFTPAR lhs RIGHTPAR
           {$$ = make_llnd(fi,ARRAY_OP,$1,$3,SMNULL);}
         | loc_op LEFTPAR variable_list COMMA expr RIGHTPAR
           {$3 = set_ll_list($3,$5,EXPR_LIST);
            $$ = make_llnd(fi,ARRAY_OP,$1,$3,SMNULL);}
         ; 

reduction_op: SUM
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "sum";
              $$->type = global_string;
             }
            | PRODUCT
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "product";
              $$->type = global_string;
             }
            | MIN
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "min";
              $$->type = global_string;
             }
            | MAX
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "max";
              $$->type = global_string;
             }
            | OR
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "or";
              $$->type = global_string;
             }
            | AND
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "and";
              $$->type = global_string;
             }
            | EQV
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "eqv";
              $$->type = global_string;
             }
            | NEQV
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "neqv";
              $$->type = global_string;
             }
            | UNKNOWN
             { err("Illegal reduction operation name", 70);
               errcnt--;
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "unknown";
              $$->type = global_string;
             }
            ;

loc_op:       MAXLOC
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "maxloc";
              $$->type = global_string;
             }
            | MINLOC
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "minloc";
              $$->type = global_string;
             }
/*
            | UNKNOWN
             { err("Illegal reduction operation name", 70);
               errcnt--; 
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "unknown";
              $$->type = global_string;
             }
*/
             ;

shadow_spec: COMMA needkeyword SHADOW_RENEW LEFTPAR shadow_list RIGHTPAR
                { $$ = make_llnd(fi,SHADOW_RENEW_OP,$5,LLNULL,SMNULL);}
           | COMMA needkeyword SHADOW_START_SPEC shadow_group_name
/*
                { PTR_LLND q;
                  q = make_llnd(fi,VAR_REF,LLNULL,LLNULL,$4);
                 $$ = make_llnd(fi,SHADOW_START_OP,q,LLNULL,SMNULL);
                }
*/
                { $$ = make_llnd(fi,SHADOW_START_OP,LLNULL,LLNULL,$4);}
           | COMMA needkeyword SHADOW_WAIT_SPEC  shadow_group_name
                
/*              {  PTR_LLND q;
                  q = make_llnd(fi,VAR_REF,LLNULL,LLNULL,$4);
                  $$ = make_llnd(fi,SHADOW_WAIT_OP,q,LLNULL,SMNULL);
                } 
*/
                { $$ = make_llnd(fi,SHADOW_WAIT_OP,LLNULL,LLNULL,$4);} 
           | COMMA needkeyword SHADOW_COMPUTE 
                { $$ = make_llnd(fi,SHADOW_COMP_OP,LLNULL,LLNULL,SMNULL);}  
           | COMMA needkeyword SHADOW_COMPUTE LEFTPAR  array_ident LEFTPAR  sh_width_list RIGHTPAR  RIGHTPAR 
                {  $5-> entry.Template.ll_ptr1 = $7; $$ = make_llnd(fi,SHADOW_COMP_OP,$5,LLNULL,SMNULL);}  
           ;

shadow_group_name:  name
       {$$ = make_local_entity($1, SHADOW_GROUP_NAME,global_default,LOCAL);}
            ;        

shadow_list: shadow
              { $$ = set_ll_list($1,LLNULL,EXPR_LIST);}
           | shadow_list COMMA shadow
              { $$ = set_ll_list($1,$3,EXPR_LIST);}
           ;

shadow: array_ident
        { $$ = $1;} /* $$->variant is ARRAY_REF */
      | array_ident LEFTPAR opt_corner CORNER  RIGHTPAR
        { PTR_LLND p;
          p = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
          p->entry.string_val = (char *) "corner";
          p->type = global_string;
          $$ = make_llnd(fi,ARRAY_OP,$1,p,SMNULL);
         }

      | array_ident LEFTPAR  opt_corner sh_width_list RIGHTPAR
        { $$ = $1;
          $$-> entry.Template.ll_ptr1 = $4;  
        }
      | array_ident LEFTPAR opt_corner sh_width_list RIGHTPAR LEFTPAR needkeyword CORNER RIGHTPAR
       { PTR_LLND p;
          p = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
          p->entry.string_val = (char *) "corner";
          p->type = global_string;
          $1-> entry.Template.ll_ptr1 = $4;  
          $$ = make_llnd(fi,ARRAY_OP,$1,p,SMNULL);
       }

      ;
 
opt_corner: 
           { optcorner = YES; }
          ;

array_ident: ident
       { PTR_SYMB s;
         s = $1->entry.Template.symbol;
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
                
         $$ = $1;
        }
           ;

array_ident_list: array_ident
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
              | array_ident_list COMMA array_ident
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;

dvm_shadow_start: SHADOW_START end_spec shadow_group_name
         { $$ = get_bfnd(fi,DVM_SHADOW_START_DIR,$3,LLNULL,LLNULL,LLNULL);} 
                |  SHADOW_START_SPEC end_spec shadow_group_name
                {errstr("Missing DVM directive prefix", 49);}
                ;

dvm_shadow_wait: SHADOW_WAIT  end_spec shadow_group_name
         { $$ = get_bfnd(fi,DVM_SHADOW_WAIT_DIR,$3,LLNULL,LLNULL,LLNULL);}  
                |  SHADOW_WAIT_SPEC end_spec shadow_group_name
                {errstr("Missing DVM directive prefix", 49);}
               ;

dvm_shadow_group: SHADOW_GROUP end_spec shadow_group_name LEFTPAR shadow_list RIGHTPAR
          { $$ = get_bfnd(fi,DVM_SHADOW_GROUP_DIR,$3,$5,LLNULL,LLNULL);}   
                ;

dvm_reduction_start: REDUCTION_START end_spec reduction_group
         { $$ = get_bfnd(fi,DVM_REDUCTION_START_DIR,$3,LLNULL,LLNULL,LLNULL);} 
                ;

dvm_reduction_wait: REDUCTION_WAIT  end_spec reduction_group
         { $$ = get_bfnd(fi,DVM_REDUCTION_WAIT_DIR,$3,LLNULL,LLNULL,LLNULL);}  
               ;
/*
dvm_reduction_group: REDUCTION_GROUP  end_spec reduction_group_name LEFTPAR reduction_list RIGHTPAR
          { $$ = get_bfnd(fi,DVM_REDUCTION_GROUP_DIR,$3,$5,LLNULL,LLNULL);}
                   ;
*/

dvm_consistent_start: CONSISTENT_START end_spec consistent_group
         { $$ = get_bfnd(fi,DVM_CONSISTENT_START_DIR,$3,LLNULL,LLNULL,LLNULL);} 
                ;

dvm_consistent_wait: CONSISTENT_WAIT  end_spec consistent_group
         { $$ = get_bfnd(fi,DVM_CONSISTENT_WAIT_DIR,$3,LLNULL,LLNULL,LLNULL);}  
               ;

dvm_remote_access: REMOTE_ACCESS end_spec LEFTPAR group_name COLON remote_data_list RIGHTPAR 
         { if(($4->attr & INDIRECT_BIT))
                errstr("'%s' is not remote group name", $4->ident, 68);
           $$ = get_bfnd(fi,DVM_REMOTE_ACCESS_DIR,$4,$6,LLNULL,LLNULL);
         }
                 | REMOTE_ACCESS  end_spec LEFTPAR remote_data_list RIGHTPAR 
         { $$ = get_bfnd(fi,DVM_REMOTE_ACCESS_DIR,SMNULL,$4,LLNULL,LLNULL);}
                 ;

group_name: name
          {  
            if(($$=$1->id_attr) == SMNULL){
                errstr("'%s' is not declared as group", $1->ident, 74);
                $$ = make_local_entity($1,REF_GROUP_NAME,global_default,LOCAL);
            } else {
              if($$->variant != REF_GROUP_NAME)
                errstr("'%s' is not declared as group", $1->ident, 74);
            }
          }
          ;

remote_data_list: remote_data
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
              | remote_data_list COMMA remote_data
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;

remote_data: array_ident LEFTPAR remote_index_list RIGHTPAR
            {
              $$ = $1;
              $$->entry.Template.ll_ptr1 = $3;
            }
           | array_ident
             { $$ = $1;}
           ;

remote_index_list: remote_index
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
                 | remote_index_list COMMA remote_index
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;

remote_index: expr
            { $$ = $1;}
            | COLON
            { $$ = make_llnd(fi,DDOT, LLNULL, LLNULL, SMNULL);}
             ;

dvm_task: TASK in_dcl  task_array
          {  PTR_LLND q;
             q = make_llnd(fi,EXPR_LIST, $3, LLNULL, SMNULL);
             $$ = get_bfnd(fi,DVM_TASK_DIR,SMNULL,q,LLNULL,LLNULL);
          }
        | dvm_task COMMA task_array
          {   PTR_LLND q;
              q = make_llnd(fi,EXPR_LIST, $3, LLNULL, SMNULL);
	      add_to_lowLevelList(q, $1->entry.Template.ll_ptr1);
          }
        ;

task_array: name dims
           { 
             PTR_SYMB s;
	      s = make_array($1, global_int, $2, ndim, LOCAL);
              if($2){
                  s->attr = s->attr | DIMENSION_BIT;
                  s->type->entry.ar_decl.ranges = $2;
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
    
	      $$ = make_llnd(fi,ARRAY_REF, $2, LLNULL, s);	  
	    }    
	;

dvm_task_region: TASK_REGION  end_spec task_name  
                 {$$ = get_bfnd(fi,DVM_TASK_REGION_DIR,$3,LLNULL,LLNULL,LLNULL);}             
               | TASK_REGION  end_spec task_name reduction_spec
                {$$ = get_bfnd(fi,DVM_TASK_REGION_DIR,$3,$4,LLNULL,LLNULL);}
               | TASK_REGION  end_spec task_name consistent_spec
                {$$ = get_bfnd(fi,DVM_TASK_REGION_DIR,$3,LLNULL,$4,LLNULL);}
               | TASK_REGION  end_spec task_name reduction_spec consistent_spec
                {$$ = get_bfnd(fi,DVM_TASK_REGION_DIR,$3,$4,$5,LLNULL);}              
               | TASK_REGION  end_spec task_name consistent_spec reduction_spec
                {$$ = get_bfnd(fi,DVM_TASK_REGION_DIR,$3,$5,$4,LLNULL);}                  
               ;

task_name:   name
             { PTR_SYMB s;
              if((s=$1->id_attr) == SMNULL)
                s = make_array($1, TYNULL, LLNULL, 0, LOCAL);
              
              if(!(s->attr & TASK_BIT))
                 errstr("'%s' is not task array", s->ident, 77);
              $$ = s;
              }
              ;

dvm_end_task_region:  ENDTASK_REGION  end_spec
          { $$ = get_bfnd(fi,DVM_END_TASK_REGION_DIR,SMNULL,LLNULL,LLNULL,LLNULL);}
                   ;

task:  task_name  LEFTPAR expr RIGHTPAR
           {  PTR_SYMB s;
              PTR_LLND q;
             /*
              s = make_array($1, TYNULL, LLNULL, 0, LOCAL);                           
	      if((parstate == INEXEC) && !(s->attr & TASK_BIT))
                 errstr("'%s' is not task array", s->ident, 77);  
              q =  set_ll_list($3,LLNULL,EXPR_LIST);
	      $$ = make_llnd(fi,ARRAY_REF, q, LLNULL, s);
              */

              s = $1;
              q =  set_ll_list($3,LLNULL,EXPR_LIST);
	      $$ = make_llnd(fi,ARRAY_REF, q, LLNULL, s);
	   }
     | task_name  LEFTPAR triplet RIGHTPAR 
           {  PTR_LLND q; 
              q =  set_ll_list($3,LLNULL,EXPR_LIST);
	      $$ = make_llnd(fi,ARRAY_REF, q, LLNULL, $1);
	   }
            ;        

dvm_on: ON_DIR end_spec array_ele_substring_func_ref opt_private_spec
    {              
         $$ = get_bfnd(fi,DVM_ON_DIR,SMNULL,$3,$4,LLNULL);
    } 
      ;  

opt_private_spec: 
             {$$ = LLNULL;}
            | private_spec
             { $$ = $1;}
            ;

dvm_end_on: ENDON end_spec
         {$$ = get_bfnd(fi,DVM_END_ON_DIR,SMNULL,LLNULL,LLNULL,LLNULL);}
          ;

dvm_map:  MAP end_spec task  needkeyword ONTO processors_name dims
      { PTR_LLND q;
        /* if(!($6->attr & PROCESSORS_BIT))
           errstr("'%s' is not processor array", $6->ident, 67);
         */
        q = make_llnd(fi,ARRAY_REF, $7, LLNULL, $6);
        $$ = get_bfnd(fi,DVM_MAP_DIR,SMNULL,$3,q,LLNULL);
      }
      | MAP end_spec task  needkeyword BY lhs
       { $$ = get_bfnd(fi,DVM_MAP_DIR,SMNULL,$3,LLNULL,$6); } 
       ;

dvm_prefetch: PREFETCH end_spec group_name
              { $$ = get_bfnd(fi,DVM_PREFETCH_DIR,$3,LLNULL,LLNULL,LLNULL);} 
            ;

dvm_reset: RESET end_spec group_name
           { $$ = get_bfnd(fi,DVM_RESET_DIR,$3,LLNULL,LLNULL,LLNULL);} 
         ;
/*
dvm_indirect_access: INDIRECT_ACCESS end_spec LEFTPAR opt_group indirect_list RIGHTPAR 
                    { $$ = get_bfnd(fi,DVM_INDIRECT_ACCESS_DIR,$4,$5,LLNULL,LLNULL);} 
                   ;
*/
dvm_indirect_access: INDIRECT_ACCESS end_spec LEFTPAR group_name COLON indirect_list RIGHTPAR 
                    { if(!($4->attr & INDIRECT_BIT))
                         errstr("'%s' is not indirect group name", $4->ident, 313);
                      $$ = get_bfnd(fi,DVM_INDIRECT_ACCESS_DIR,$4,$6,LLNULL,LLNULL);
                    }
                   | INDIRECT_ACCESS end_spec LEFTPAR  indirect_list RIGHTPAR 
                    { $$ = get_bfnd(fi,DVM_INDIRECT_ACCESS_DIR,SMNULL,$4,LLNULL,LLNULL);} 
                   ;
/*
opt_group:
               { $$ = SMNULL;}
         |  group_name COLON 
               { if(!($1->attr & INDIRECT_BIT))
                   errstr("'%s' is not indirect group name", $1->ident, 313);
                 $$ = $1;
                }
         ;
*/

indirect_list: indirect_reference
	    { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
            | indirect_list COMMA  indirect_reference 
	    { $$ = set_ll_list($1,$3,EXPR_LIST); }	    
	;

indirect_reference: array_ident
                    { $$ = $1;} 
                  | array_ident LEFTPAR subscript_list RIGHTPAR
                    { $$ = $1; $$->entry.Template.ll_ptr1 = $3;} 
                  ; 
/* 
dvm_own: OWN
          { $$ = get_bfnd(fi,DVM_OWN_DIR,SMNULL,LLNULL,LLNULL,LLNULL);} 
       ;
*/
  
hpf_independent: INDEPENDENT end_spec
                { $$ = get_bfnd(fi,HPF_INDEPENDENT_DIR,SMNULL,LLNULL,LLNULL,LLNULL);}  
               | INDEPENDENT end_spec new_spec
                { $$ = get_bfnd(fi,HPF_INDEPENDENT_DIR,SMNULL, $3, LLNULL, LLNULL);}
               | INDEPENDENT end_spec hpf_reduction_spec
                { $$ = get_bfnd(fi,HPF_INDEPENDENT_DIR,SMNULL, LLNULL, $3, LLNULL);}
               | INDEPENDENT end_spec new_spec hpf_reduction_spec
                { $$ = get_bfnd(fi,HPF_INDEPENDENT_DIR,SMNULL, $3, $4,LLNULL);}

               ;
 /*
               | INDEPENDENT end_spec COMMA needkeyword NEW LEFTPAR new_variable_list  RIGHTPAR
      { PTR_LLND p;
        p  = make_llnd(fi,NEW_SPEC_OP, $7, LLNULL, SMNULL);
        $$ = get_bfnd(fi,HPF_INDEPENDENT_DIR,SMNULL, p, LLNULL,LLNULL);
      }  
  */      

/*
new_variable_list: new_variable
            { $$ = set_ll_list($1,LLNULL,EXPR_LIST);}
             | new_variable_list COMMA new_variable
            { $$ = set_ll_list($1,$3,EXPR_LIST);}
             ;

new_variable: lhs
        { PTR_SYMB s;
          if($1->variant == FUNC_CALL)
               errstr("illegal function call");
          s = $1->entry.Template.symbol;
          if(s->attr & SAVE_BIT)
               errstr("'%s' has the SAVE attribute", s->ident);  
          if(s->attr & TARGET_BIT)
                 errstr("'%s' has the TARGET attribute", s->ident);
          if(s->attr & DVM_POINTER_BIT)
                 errstr("'%s' is a pointer", s->ident);
          if(s->entry.var_decl.local == IO)
                 errstr("'%s' is a dummy argument", s->ident);
        }
           ;
*/

hpf_reduction_spec: COMMA needkeyword REDUCTION LEFTPAR variable_list RIGHTPAR
              {$$ = make_llnd(fi,REDUCTION_OP,$5,LLNULL,SMNULL);}
                  ;

dvm_asynchronous: ASYNCHRONOUS end_spec async
                { $$ = get_bfnd(fi,DVM_ASYNCHRONOUS_DIR,SMNULL,$3,LLNULL,LLNULL);} 
                ;

dvm_endasynchronous: ENDASYNCHRONOUS end_spec 
                { $$ = get_bfnd(fi,DVM_ENDASYNCHRONOUS_DIR,SMNULL,LLNULL,LLNULL,LLNULL);} 
                ;

dvm_asyncwait: ASYNCWAIT end_spec async
                { $$ = get_bfnd(fi,DVM_ASYNCWAIT_DIR,SMNULL,$3,LLNULL,LLNULL);} 
             ;

async_ident: name 
     {  
            if(($$=$1->id_attr) == SMNULL) {
                errstr("'%s' is not declared as ASYNCID", $1->ident, 115);
                $$ = make_local_entity($1,ASYNC_ID,global_default,LOCAL);
            } else {
              if($$->variant != ASYNC_ID)
                errstr("'%s' is not declared as ASYNCID", $1->ident, 115);
            }
     }
     ;

async: async_ident
      { $$ = make_llnd(fi,VAR_REF, LLNULL, LLNULL, $1);}
     | async_ident LEFTPAR subscript_list RIGHTPAR
       { $$ = make_llnd(fi,ARRAY_REF, $3, LLNULL, $1);} 
     ;

dvm_f90: F90 end_spec lhs EQUAL lhs
        { $$ = get_bfnd(fi,DVM_F90_DIR,SMNULL,$3,$5,LLNULL);} 
       ;
dvm_debug_dir: DEBUG end_spec fragment_number
               { $$ = get_bfnd(fi,DVM_DEBUG_DIR,SMNULL,$3,LLNULL,LLNULL);} 
             |  DEBUG end_spec fragment_number LEFTPAR debparamlist RIGHTPAR 
               { $$ = get_bfnd(fi,DVM_DEBUG_DIR,SMNULL,$3,$5,LLNULL);} 
             ;

debparamlist:  in_ioctl debparam
	    { 
              $$ = set_ll_list($2, LLNULL, EXPR_LIST);
              endioctl();
            }
	    | debparamlist COMMA in_ioctl debparam
	    { 
              $$ = set_ll_list($1, $4, EXPR_LIST);
              endioctl();
            }
	    ;

debparam:  nameeq expr
           { $$  = make_llnd(fi, KEYWORD_ARG, $1, $2, SMNULL); } /* SPEC_PAIR*/
        ;
fragment_number:  INT_CONSTANT
	        {
	         $$ = make_llnd(fi,INT_VAL, LLNULL, LLNULL, SMNULL);
	         $$->entry.ival = atoi(yytext);
	         $$->type = global_int;
	        }
              ;

dvm_enddebug_dir: ENDDEBUG end_spec fragment_number
       { $$ = get_bfnd(fi,DVM_ENDDEBUG_DIR,SMNULL,$3,LLNULL,LLNULL);} 
             ;

dvm_interval_dir: INTERVAL end_spec interval_number 
       { $$ = get_bfnd(fi,DVM_INTERVAL_DIR,SMNULL,$3,LLNULL,LLNULL);} 
             ;

interval_number:
                { $$ = LLNULL;}

               |  expr
                { if($1->type->variant != T_INT)             
                    err("Illegal interval number", 78);
                  $$ = $1;
                 }
               ;

dvm_exit_interval_dir: EXITINTERVAL end_spec subscript_list 
                      /* subscript_list - interval number list */ 
       { $$ = get_bfnd(fi,DVM_EXIT_INTERVAL_DIR,SMNULL,$3,LLNULL,LLNULL);} 
             ;

dvm_endinterval_dir: ENDINTERVAL end_spec
            { $$ = get_bfnd(fi,DVM_ENDINTERVAL_DIR,SMNULL,LLNULL,LLNULL,LLNULL);} 
                   ;

dvm_traceon_dir: TRACEON end_spec
            { $$ = get_bfnd(fi,DVM_TRACEON_DIR,SMNULL,LLNULL,LLNULL,LLNULL);} 
               ;

dvm_traceoff_dir: TRACEOFF end_spec
            { $$ = get_bfnd(fi,DVM_TRACEOFF_DIR,SMNULL,LLNULL,LLNULL,LLNULL);} 
               ;

dvm_barrier_dir: BARRIER end_spec
            { $$ = get_bfnd(fi,DVM_BARRIER_DIR,SMNULL,LLNULL,LLNULL,LLNULL);} 
               ;

dvm_check: CHECK end_spec in_ioctl LEFTPAR callarglist RIGHTPAR end_ioctl opt_double_colon variable_list 
          { $$ = get_bfnd(fi,DVM_CHECK_DIR,SMNULL,$9,$5,LLNULL); }         
         ;

dvm_io_mode_dir: IO_MODE end_spec LEFTPAR mode_list RIGHTPAR
            { $$ = get_bfnd(fi,DVM_IO_MODE_DIR,SMNULL,$4,LLNULL,LLNULL);} 
               ;
mode_list:   mode_spec
	       { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
         |   mode_list COMMA mode_spec
	       { $$ = set_ll_list($1,$3,EXPR_LIST); }	
         ;

mode_spec:  needkeyword ACC_ASYNC
              { $$ = make_llnd(fi,ACC_ASYNC_OP,LLNULL,LLNULL,SMNULL);}
         |  needkeyword ACC_LOCAL
              { $$ = make_llnd(fi,ACC_LOCAL_OP, LLNULL,LLNULL,SMNULL);}
         |  needkeyword PARALLEL
              { $$ = make_llnd(fi,PARALLEL_OP, LLNULL,LLNULL,SMNULL);}
         ;

dvm_shadow_add: SHADOW_ADD end_spec LEFTPAR template_ref EQUAL shadow_id RIGHTPAR opt_key_word opt_include_to 
              { $$ = get_bfnd(fi,DVM_SHADOW_ADD_DIR,SMNULL,$4,$6,$9); }
              ;

template_ref:  ident LEFTPAR shadow_axis_list RIGHTPAR
               {
                 if($1->type->variant != T_ARRAY) 
                    errstr("'%s' isn't array", $1->entry.Template.symbol->ident, 66);
                 if(!($1->entry.Template.symbol->attr & TEMPLATE_BIT))
                    errstr("'%s' isn't TEMPLATE", $1->entry.Template.symbol->ident, 628);
                 $1->entry.Template.ll_ptr1 = $3;
                 $$ = $1;
                 /*$$->type = $1->type->entry.ar_decl.base_type;*/
               }
            ;

shadow_axis_list: shadow_axis
	          { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
                | shadow_axis_list COMMA shadow_axis
	          { $$ = set_ll_list($1,$3,EXPR_LIST); }	
                ;

shadow_axis: derived_spec
             { $$ = $1; }
           | aster_expr
             { $$ = $1; }
           ;

opt_include_to: INCLUDE_TO array_ident_list
                { $$ = $2;}              
	      |  
                { $$ = LLNULL; opt_kwd_ = NO;}
              ;
 
dvm_localize: LOCALIZE end_spec LEFTPAR ident POINT_TO localize_target  RIGHTPAR
              { $$ = get_bfnd(fi,DVM_LOCALIZE_DIR,SMNULL,$4,$6,LLNULL); }
            ;

localize_target: ident
                {
                 if($1->type->variant != T_ARRAY) 
                    errstr("'%s' isn't array", $1->entry.Template.symbol->ident, 66); 
                 $$ = $1;
                }
               | ident  LEFTPAR target_subscript_list RIGHTPAR
                {
                 if($1->type->variant != T_ARRAY) 
                    errstr("'%s' isn't array", $1->entry.Template.symbol->ident, 66); 
                                 
                 $1->entry.Template.ll_ptr1 = $3;
                 $$ = $1;
                 $$->type = $1->type->entry.ar_decl.base_type;
                }
                                                          
               ;

target_subscript_list: target_subscript
	               { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
                     | target_subscript_list COMMA target_subscript
	               { $$ = set_ll_list($1,$3,EXPR_LIST); }	
                     ;

target_subscript: aster_expr
                { $$ = $1; }
                | COLON
                { $$ = make_llnd(fi,DDOT, LLNULL, LLNULL, SMNULL);}
                ;

aster_expr: ASTER
          { 
            $$ = make_llnd(fi,KEYWORD_VAL, LLNULL, LLNULL, SMNULL);
            $$->entry.string_val = (char *) "*";
            $$->type = global_string;
          }
          ;

dvm_cp_create: CP_CREATE end_spec expr COMMA needkeyword VARLIST LEFTPAR variable_list RIGHTPAR COMMA needkeyword FILES LEFTPAR subscript_list RIGHTPAR  opt_mode 
              { 
                PTR_LLND q;
                if($16)
                  q = make_llnd(fi,ARRAY_OP, $14, $16, SMNULL);
                else
                  q = $14;                  
                $$ = get_bfnd(fi,DVM_CP_CREATE_DIR,SMNULL,$3,$8,q); 
              } 
             ;

opt_mode:
              { $$ = LLNULL; }
            |  COMMA needkeyword PARALLEL
              { $$ = make_llnd(fi, PARALLEL_OP, LLNULL, LLNULL, SMNULL); }
            |  COMMA needkeyword ACC_LOCAL 
              { $$ = make_llnd(fi,ACC_LOCAL_OP, LLNULL, LLNULL, SMNULL); }  
            ;

dvm_cp_load:   CP_LOAD end_spec expr
              { $$ = get_bfnd(fi,DVM_CP_LOAD_DIR,SMNULL,$3,LLNULL,LLNULL); } 
           ;

dvm_cp_save:   CP_SAVE end_spec expr
              { $$ = get_bfnd(fi,DVM_CP_SAVE_DIR,SMNULL,$3,LLNULL,LLNULL); }              
           |   CP_SAVE end_spec expr COMMA needkeyword ACC_ASYNC 
              {
                PTR_LLND q;
                q = make_llnd(fi,ACC_ASYNC_OP,LLNULL,LLNULL,SMNULL);
                $$ = get_bfnd(fi,DVM_CP_SAVE_DIR,SMNULL,$3,q,LLNULL);
              }
           ;

dvm_cp_wait:   CP_WAIT end_spec expr COMMA needkeyword STATUS LEFTPAR ident RIGHTPAR
              { $$ = get_bfnd(fi,DVM_CP_WAIT_DIR,SMNULL,$3,$8,LLNULL); }
           ;

dvm_template_create: TEMPLATE_CREATE end_spec  LEFTPAR  template_list RIGHTPAR
                     { $$ = get_bfnd(fi,DVM_TEMPLATE_CREATE_DIR,SMNULL,$4,LLNULL,LLNULL); }
                   ;
template_list: array_element
	       { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	     | template_list COMMA array_element
               { $$ = set_ll_list($1, $3, EXPR_LIST); }
             ;

dvm_template_delete: TEMPLATE_DELETE end_spec  LEFTPAR ident_list RIGHTPAR
                     { $$ = get_bfnd(fi,DVM_TEMPLATE_DELETE_DIR,SMNULL,$4,LLNULL,LLNULL); }
                   ;

omp_specification_directive: omp_threadprivate_directive
	;
omp_execution_directive: omp_parallel_begin_directive
	| omp_parallel_end_directive
	| omp_sections_begin_directive
	| omp_sections_end_directive
	| omp_section_directive
	| omp_do_begin_directive
	| omp_do_end_directive
	| omp_single_begin_directive
	| omp_single_end_directive
	| omp_workshare_begin_directive
	| omp_workshare_end_directive
	| omp_parallel_do_begin_directive
	| omp_parallel_do_end_directive
	| omp_parallel_sections_begin_directive
	| omp_parallel_sections_end_directive
	| omp_parallel_workshare_begin_directive
	| omp_parallel_workshare_end_directive
	| omp_master_begin_directive
	| omp_master_end_directive
	| omp_ordered_begin_directive
	| omp_ordered_end_directive
	| omp_barrier_directive
	| omp_atomic_directive
	| omp_flush_directive
	| omp_critical_begin_directive
	| omp_critical_end_directive
	;

ompdvm_onethread: OMPDVM_ONETHREAD end_spec 
	{
          $$ = get_bfnd(fi,OMP_ONETHREAD_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	};


omp_parallel_end_directive: OMPDVM_ENDPARALLEL end_spec 
	{
  	   $$ = make_endparallel();
	};


omp_parallel_begin_directive: OMPDVM_PARALLEL end_spec opt_key_word parallel_clause_list
	{
  	   $$ = make_parallel();
           $$->entry.Template.ll_ptr1 = $4;
	   opt_kwd_ = NO;
	}
	| OMPDVM_PARALLEL end_spec opt_key_word
	{
  	   $$ = make_parallel();
	   opt_kwd_ = NO;
	};

parallel_clause_list: opt_comma opt_key_word parallel_clause opt_key_word
	{ 
		$$ = set_ll_list($3,LLNULL,EXPR_LIST);
	}
	| parallel_clause_list opt_comma opt_key_word parallel_clause opt_key_word
	{ 
		$$ = set_ll_list($1,$4,EXPR_LIST);	
	}
	;

parallel_clause:  ompprivate_clause
        | ompreduction_clause
        | ompshared_clause
        | ompdefault_clause
        | ompfirstprivate_clause
	| omplastprivate_clause
        | ompcopyin_clause
	| ompif_clause
	| ompnumthreads_clause
        ;

omp_variable_list_in_par: op_slash_1 LEFTPAR op_slash_0 omp_variable_list RIGHTPAR
        {
		$$ = $4;
        };

ompprivate_clause: PRIVATE omp_variable_list_in_par
	{
		$$ = make_llnd(fi,OMP_PRIVATE,$2,LLNULL,SMNULL);
	};

ompfirstprivate_clause: OMPDVM_FIRSTPRIVATE omp_variable_list_in_par
	{
		$$ = make_llnd(fi,OMP_FIRSTPRIVATE,$2,LLNULL,SMNULL);
	}
        ;

omplastprivate_clause: OMPDVM_LASTPRIVATE omp_variable_list_in_par
	{
		$$ = make_llnd(fi,OMP_LASTPRIVATE,$2,LLNULL,SMNULL);
	}
        ;

ompcopyin_clause: OMPDVM_COPYIN omp_variable_list_in_par
	{
		$$ = make_llnd(fi,OMP_COPYIN,$2,LLNULL,SMNULL);
	}
        ;

ompshared_clause: OMPDVM_SHARED omp_variable_list_in_par
	{
		$$ = make_llnd(fi,OMP_SHARED,$2,LLNULL,SMNULL);
	}
        ;
ompdefault_clause: DEFAULT_CASE LEFTPAR needkeyword def_expr RIGHTPAR
	{
		$$ = make_llnd(fi,OMP_DEFAULT,$4,LLNULL,SMNULL);
	}
        ;

def_expr: PRIVATE
	{
		$$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		$$->entry.string_val = (char *) "private";
		$$->type = global_string;
	}
	| OMPDVM_SHARED
	{
		$$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		$$->entry.string_val = (char *) "shared";
		$$->type = global_string;
	}
	| OMPDVM_NONE
	{
		$$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		$$->entry.string_val = (char *) "none";
		$$->type = global_string;
	}
	;
ompif_clause: OMPDVM_IF LEFTPAR expr RIGHTPAR
	{
		$$ = make_llnd(fi,OMP_IF,$3,LLNULL,SMNULL);
	}
        ;

ompnumthreads_clause: OMPDVM_NUM_THREADS LEFTPAR expr RIGHTPAR
	{
		$$ = make_llnd(fi,OMP_NUM_THREADS,$3,LLNULL,SMNULL);
	}
        ;

ompreduction_clause: REDUCTION LEFTPAR ompreduction RIGHTPAR
	{
		PTR_LLND q;
		q = set_ll_list($3,LLNULL,EXPR_LIST);
		$$ = make_llnd(fi,OMP_REDUCTION,q,LLNULL,SMNULL);
	};

ompreduction:  opt_key_word ompreduction_op COLON ompreduction_vars
           {$$ = make_llnd(fi,DDOT,$2,$4,SMNULL);}
         ; 

ompreduction_vars: variable_list;

ompreduction_op: PLUS
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "+";
              $$->type = global_string;
             }
	    | MINUS
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "-";
              $$->type = global_string;
             }

            | ASTER
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "*";
              $$->type = global_string;
             }
            | SLASH
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "/";
              $$->type = global_string;
             }
            | MIN
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "min";
              $$->type = global_string;
             }
            | MAX
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "max";
              $$->type = global_string;
             }
            | OR
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) ".or.";
              $$->type = global_string;
             }
            | AND
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) ".and.";
              $$->type = global_string;
             }
            | EQV
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) ".eqv.";
              $$->type = global_string;
             }
            | NEQV
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) ".neqv.";
              $$->type = global_string;
             }
            | IAND
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "iand";
              $$->type = global_string;
             }
            | IEOR
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "ieor";
              $$->type = global_string;
             }
            | IOR
             {
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "ior";
              $$->type = global_string;
             }
            | UNKNOWN
             { err("Illegal reduction operation name", 70);
               errcnt--;
              $$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
              $$->entry.string_val = (char *) "unknown";
              $$->type = global_string;
             }
            ;


omp_sections_begin_directive: OMPDVM_SECTIONS end_spec opt_key_word sections_clause_list 
	{
  	   $$ = make_sections($4);
	   opt_kwd_ = NO;
	}
	| OMPDVM_SECTIONS end_spec opt_key_word
	{
  	   $$ = make_sections(LLNULL);
	   opt_kwd_ = NO;
	};

sections_clause_list: opt_comma opt_key_word sections_clause opt_key_word
	{ 
		$$ = set_ll_list($3,LLNULL,EXPR_LIST);
	}
	| sections_clause_list opt_comma opt_key_word sections_clause opt_key_word
	{ 
		$$ = set_ll_list($1,$4,EXPR_LIST);
	}
	;

sections_clause:  ompprivate_clause
        | ompreduction_clause
        | ompfirstprivate_clause
	| omplastprivate_clause
        ;

omp_sections_end_directive: OMPDVM_ENDSECTIONS end_spec opt_key_word ompnowait_clause
	{
		PTR_LLND q;
   	        $$ = make_endsections();
		q = set_ll_list($4,LLNULL,EXPR_LIST);
                $$->entry.Template.ll_ptr1 = q;
                opt_kwd_ = NO;
	}
	| OMPDVM_ENDSECTIONS end_spec opt_key_word
	{
   	        $$ = make_endsections();
	        opt_kwd_ = NO; 
	};

omp_section_directive: OMPDVM_SECTION end_spec
	{
           $$ = make_ompsection();
	};


omp_do_begin_directive: OMPDVM_DO end_spec opt_key_word do_clause_list 
	{
           $$ = get_bfnd(fi,OMP_DO_DIR,SMNULL,$4,LLNULL,LLNULL);
	   opt_kwd_ = NO;
	}
	| OMPDVM_DO end_spec opt_key_word
	{
           $$ = get_bfnd(fi,OMP_DO_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	   opt_kwd_ = NO;
	};

omp_do_end_directive: OMPDVM_ENDDO end_spec opt_key_word ompnowait_clause
	{
		PTR_LLND q;
		q = set_ll_list($4,LLNULL,EXPR_LIST);
	        $$ = get_bfnd(fi,OMP_END_DO_DIR,SMNULL,q,LLNULL,LLNULL);
      	        opt_kwd_ = NO;
	}
	| OMPDVM_ENDDO end_spec opt_key_word 
	{
           $$ = get_bfnd(fi,OMP_END_DO_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	   opt_kwd_ = NO;
	};

do_clause_list: opt_comma opt_key_word do_clause opt_key_word
	{ 
		$$ = set_ll_list($3,LLNULL,EXPR_LIST);
	}
	| do_clause_list opt_comma opt_key_word do_clause opt_key_word
	{ 
		$$ = set_ll_list($1,$4,EXPR_LIST);
	}
	;

do_clause:  ompprivate_clause
        | ompreduction_clause
        | ompfirstprivate_clause
	| omplastprivate_clause
	| ompschedule_clause
	| ompordered_clause
        ;

ompordered_clause: OMPDVM_ORDERED
	{
		/*$$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		$$->entry.string_val = (char *) "ORDERED";
		$$->type = global_string;*/
                $$ = make_llnd(fi,OMP_ORDERED,LLNULL,LLNULL,SMNULL);
	}
	;

ompschedule_clause: OMPDVM_SCHEDULE LEFTPAR needkeyword ompschedule_op COMMA expr RIGHTPAR
	{
		$$ = make_llnd(fi,OMP_SCHEDULE,$4,$6,SMNULL);
	}
	| OMPDVM_SCHEDULE LEFTPAR needkeyword ompschedule_op RIGHTPAR
	{
		$$ = make_llnd(fi,OMP_SCHEDULE,$4,LLNULL,SMNULL);
	}
	;

ompschedule_op: STATIC
	{
		$$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		$$->entry.string_val = (char *) "STATIC";
		$$->type = global_string;
		
	}
	| DYNAMIC
	{
		$$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		$$->entry.string_val = (char *) "DYNAMIC";
		$$->type = global_string;
		
	}
	| OMPDVM_GUIDED
	{
		$$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		$$->entry.string_val = (char *) "GUIDED";
		$$->type = global_string;
		
	}
	| OMPDVM_RUNTIME
	{
		$$ = make_llnd(fi,KEYWORD_VAL,LLNULL,LLNULL,SMNULL);
		$$->entry.string_val = (char *) "RUNTIME";
		$$->type = global_string;
		
	}
	;

omp_single_begin_directive: OMPDVM_SINGLE end_spec opt_key_word single_clause_list 
	{
  	   $$ = make_single();
           $$->entry.Template.ll_ptr1 = $4;
	   opt_kwd_ = NO;
	}
	| OMPDVM_SINGLE end_spec opt_key_word
	{
  	   $$ = make_single();
	   opt_kwd_ = NO;
	};

single_clause_list: opt_comma opt_key_word single_clause opt_key_word
	{ 
		$$ = set_ll_list($3,LLNULL,EXPR_LIST);
	}
	| single_clause_list opt_comma opt_key_word single_clause opt_key_word
	{ 
		$$ = set_ll_list($1,$4,EXPR_LIST);
	}
	;

single_clause:  ompprivate_clause
        | ompfirstprivate_clause
        ;

omp_single_end_directive: OMPDVM_ENDSINGLE end_spec opt_key_word end_single_clause_list 
	{
  	   $$ = make_endsingle();
           $$->entry.Template.ll_ptr1 = $4;
	   opt_kwd_ = NO;
	}
	| OMPDVM_ENDSINGLE end_spec opt_key_word
	{
  	   $$ = make_endsingle();
	   opt_kwd_ = NO;
	};

end_single_clause_list: opt_comma opt_key_word end_single_clause opt_key_word
	{ 
		$$ = set_ll_list($3,LLNULL,EXPR_LIST);
	}
	| end_single_clause_list opt_comma opt_key_word end_single_clause opt_key_word
	{ 
		$$ = set_ll_list($1,$4,EXPR_LIST);
	}
	;


end_single_clause:  ompnowait_clause
        | ompcopyprivate_clause
        ;

ompcopyprivate_clause: OMPDVM_COPYPRIVATE omp_variable_list_in_par
	{
		$$ = make_llnd(fi,OMP_COPYPRIVATE,$2,LLNULL,SMNULL);
	}
        ;

ompnowait_clause: OMPDVM_NOWAIT
	{
		$$ = make_llnd(fi,OMP_NOWAIT,LLNULL,LLNULL,SMNULL);
	}
        ;

omp_workshare_begin_directive: OMPDVM_WORKSHARE end_spec
	{
           $$ = make_workshare();
	};

omp_workshare_end_directive: OMPDVM_ENDWORKSHARE end_spec opt_key_word ompnowait_clause
	{
		PTR_LLND q;
   	        $$ = make_endworkshare();
		q = set_ll_list($4,LLNULL,EXPR_LIST);
                $$->entry.Template.ll_ptr1 = q;
  	        opt_kwd_ = NO;
	}
	| OMPDVM_ENDWORKSHARE end_spec opt_key_word
	{
   	        $$ = make_endworkshare();
                opt_kwd_ = NO;
	};

omp_parallel_do_begin_directive: OMPDVM_PARALLELDO end_spec opt_key_word paralleldo_clause_list  
	{
           $$ = get_bfnd(fi,OMP_PARALLEL_DO_DIR,SMNULL,$4,LLNULL,LLNULL);
	   opt_kwd_ = NO;
	}
	| OMPDVM_PARALLELDO end_spec opt_key_word
	{
           $$ = get_bfnd(fi,OMP_PARALLEL_DO_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	   opt_kwd_ = NO;
	}
	;

paralleldo_clause_list: opt_comma opt_key_word paralleldo_clause opt_key_word
	{ 
		$$ = set_ll_list($3,LLNULL,EXPR_LIST);
	}
	| paralleldo_clause_list opt_comma opt_key_word paralleldo_clause opt_key_word
	{ 
		$$ = set_ll_list($1,$4,EXPR_LIST);
	}
	;

paralleldo_clause: ompprivate_clause
        | ompreduction_clause
        | ompshared_clause
        | ompdefault_clause
        | ompfirstprivate_clause
	| omplastprivate_clause
        | ompcopyin_clause
	| ompif_clause
	| ompnumthreads_clause
	| ompschedule_clause
	| ompordered_clause
        ;


omp_parallel_do_end_directive: OMPDVM_ENDPARALLELDO end_spec
	{
           $$ = get_bfnd(fi,OMP_END_PARALLEL_DO_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	};

omp_parallel_sections_begin_directive: OMPDVM_PARALLELSECTIONS end_spec opt_key_word parallel_clause_list
	{
           $$ = make_parallelsections($4);
	   opt_kwd_ = NO;
	}
	| OMPDVM_PARALLELSECTIONS end_spec opt_key_word
	{
           $$ = make_parallelsections(LLNULL);
	   opt_kwd_ = NO;
	};


omp_parallel_sections_end_directive: OMPDVM_ENDPARALLELSECTIONS end_spec
	{
           $$ = make_endparallelsections();
	};

omp_parallel_workshare_begin_directive: OMPDVM_PARALLELWORKSHARE end_spec opt_key_word parallel_clause_list 
	{
           $$ = make_parallelworkshare();
           $$->entry.Template.ll_ptr1 = $4;
	   opt_kwd_ = NO;
	}
	| OMPDVM_PARALLELWORKSHARE end_spec opt_key_word
	{
           $$ = make_parallelworkshare();
	   opt_kwd_ = NO;
	};

omp_parallel_workshare_end_directive: OMPDVM_ENDPARALLELWORKSHARE end_spec
	{
           $$ = make_endparallelworkshare();
	};

omp_threadprivate_directive: OMPDVM_THREADPRIVATE in_dcl omp_variable_list_in_par
	{ 
	   $$ = get_bfnd(fi,OMP_THREADPRIVATE_DIR, SMNULL, $3, LLNULL, LLNULL);
	};

omp_master_begin_directive: OMPDVM_MASTER end_spec 
	{
  	   $$ = make_master();
	};

omp_master_end_directive: OMPDVM_ENDMASTER end_spec 
	{
  	   $$ = make_endmaster();
	};
omp_ordered_begin_directive: OMPDVM_ORDERED end_spec 
	{
  	   $$ = make_ordered();
	};

omp_ordered_end_directive: OMPDVM_ENDORDERED end_spec 
	{
  	   $$ = make_endordered();
	};

omp_barrier_directive: OMPDVM_BARRIER end_spec 
	{
           $$ = get_bfnd(fi,OMP_BARRIER_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	};
omp_atomic_directive: OMPDVM_ATOMIC end_spec 
	{
           $$ = get_bfnd(fi,OMP_ATOMIC_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	};

omp_flush_directive: OMPDVM_FLUSH end_spec omp_variable_list_in_par
	{
           $$ = get_bfnd(fi,OMP_FLUSH_DIR,SMNULL,$3,LLNULL,LLNULL);
	}
	| OMPDVM_FLUSH end_spec 
	{
           $$ = get_bfnd(fi,OMP_FLUSH_DIR,SMNULL,LLNULL,LLNULL,LLNULL);
	}
	;

omp_critical_begin_directive: OMPDVM_CRITICAL end_spec LEFTPAR ident RIGHTPAR
	{
  	   $$ = make_critical();
           $$->entry.Template.ll_ptr1 = $4;
	}
	| OMPDVM_CRITICAL end_spec 
	{
  	   $$ = make_critical();
	}
	;

omp_critical_end_directive: OMPDVM_ENDCRITICAL end_spec LEFTPAR ident RIGHTPAR
	{
  	   $$ = make_endcritical();
           $$->entry.Template.ll_ptr1 = $4;
	}
	| OMPDVM_ENDCRITICAL end_spec 
	{
  	   $$ = make_endcritical();
	}
	;

omp_common_var: SLASH name op_slash_1 SLASH op_slash_0
	{ 
		PTR_SYMB s;
		PTR_LLND l;
		s = make_common($2); 
		l = make_llnd(fi,VAR_REF, LLNULL, LLNULL, s);
		$$ = make_llnd(fi,OMP_THREADPRIVATE, l, LLNULL, SMNULL);
	};


omp_variable_list: omp_common_var
	{
		$$ = set_ll_list($1,LLNULL,EXPR_LIST);
	}
	| ident
	{	
		$$ = set_ll_list($1,LLNULL,EXPR_LIST);
	}
	| omp_variable_list COMMA omp_common_var
	{ 
		$$ = set_ll_list($1,$3,EXPR_LIST);
	}
	| omp_variable_list COMMA ident
	{ 
		$$ = set_ll_list($1,$3,EXPR_LIST);
	}
	;

op_slash_1 : {
		operator_slash = 1;
	};
op_slash_0 : {
		operator_slash = 0;
	};

acc_directive: acc_region
             | acc_end_region      
             | acc_checksection
             | acc_end_checksection
             | acc_get_actual
             | acc_actual 
             | acc_routine
	     ;

acc_region: ACC_REGION end_spec  opt_clause 
             {  $$ = get_bfnd(fi,ACC_REGION_DIR,SMNULL,$3,LLNULL,LLNULL);}
             ;

acc_checksection: ACC_CHECKSECTION end_spec 
             {  $$ = get_bfnd(fi,ACC_CHECKSECTION_DIR,SMNULL,LLNULL,LLNULL,LLNULL);}
             ;

acc_get_actual:  ACC_GET_ACTUAL end_spec LEFTPAR acc_var_list RIGHTPAR
             {  $$ = get_bfnd(fi,ACC_GET_ACTUAL_DIR,SMNULL,$4,LLNULL,LLNULL);}
              |  ACC_GET_ACTUAL end_spec LEFTPAR RIGHTPAR
             {  $$ = get_bfnd(fi,ACC_GET_ACTUAL_DIR,SMNULL,LLNULL,LLNULL,LLNULL);}
              |  ACC_GET_ACTUAL end_spec 
             {  $$ = get_bfnd(fi,ACC_GET_ACTUAL_DIR,SMNULL,LLNULL,LLNULL,LLNULL);}
             ;

acc_actual:     ACC_ACTUAL end_spec LEFTPAR acc_var_list RIGHTPAR
             {  $$ = get_bfnd(fi,ACC_ACTUAL_DIR,SMNULL,$4,LLNULL,LLNULL);}
          |   ACC_ACTUAL end_spec LEFTPAR RIGHTPAR
             {  $$ = get_bfnd(fi,ACC_ACTUAL_DIR,SMNULL,LLNULL,LLNULL,LLNULL);}
          |   ACC_ACTUAL end_spec 
             {  $$ = get_bfnd(fi,ACC_ACTUAL_DIR,SMNULL,LLNULL,LLNULL,LLNULL);}
             ;

opt_clause:  needkeyword keywordoff
              { $$ = LLNULL;}
          |  acc_clause_list
              { $$ = $1; }
          ;

acc_clause_list:   acc_clause
	          { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
               | acc_clause_list COMMA acc_clause
	          { $$ = set_ll_list($1,$3,EXPR_LIST); }	
               ;

acc_clause:    needkeyword data_clause
               { $$ = $2;}

             | needkeyword async_clause
               { $$ = $2;}
               
             | needkeyword targets_clause
               { $$ = $2;}                   
             ;


data_clause:   INOUT   LEFTPAR acc_var_list RIGHTPAR
               { $$ = make_llnd(fi,ACC_INOUT_OP,$3,LLNULL,SMNULL);}
             | IN      LEFTPAR acc_var_list RIGHTPAR
               { $$ = make_llnd(fi,ACC_IN_OP,$3,LLNULL,SMNULL);}
             | OUT     LEFTPAR acc_var_list RIGHTPAR
               { $$ = make_llnd(fi,ACC_OUT_OP,$3,LLNULL,SMNULL);}
             | ACC_LOCAL   LEFTPAR acc_var_list RIGHTPAR
               { $$ = make_llnd(fi,ACC_LOCAL_OP,$3,LLNULL,SMNULL);}             
             | ACC_INLOCAL LEFTPAR acc_var_list RIGHTPAR
               { $$ = make_llnd(fi,ACC_INLOCAL_OP,$3,LLNULL,SMNULL);}             
             ;

targets_clause:   ACC_TARGETS LEFTPAR computer_list RIGHTPAR
               { $$ = make_llnd(fi,ACC_TARGETS_OP,$3,LLNULL,SMNULL);}
             ;

async_clause:   ACC_ASYNC             
               { $$ = make_llnd(fi,ACC_ASYNC_OP,LLNULL,LLNULL,SMNULL);}
             ;


acc_var_list: variable_list
              { $$ = $1;}
            ;

computer_list:   computer
	       { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
             | computer_list COMMA computer
	       { $$ = set_ll_list($1,$3,EXPR_LIST); }	
             ;

computer:       needkeyword ACC_HOST
              { $$ = make_llnd(fi,ACC_HOST_OP, LLNULL,LLNULL,SMNULL);}
             |  needkeyword ACC_CUDA
              { $$ = make_llnd(fi,ACC_CUDA_OP, LLNULL,LLNULL,SMNULL);}
             ;

acc_end_region: ACC_END_REGION
              {  $$ = get_bfnd(fi,ACC_END_REGION_DIR,SMNULL,LLNULL,LLNULL,LLNULL);}
              ;
     
acc_end_checksection: ACC_END_CHECKSECTION
              {  $$ = get_bfnd(fi,ACC_END_CHECKSECTION_DIR,SMNULL,LLNULL,LLNULL,LLNULL);}
              ;     
        
acc_routine: ACC_ROUTINE in_dcl opt_targets_clause
             {  $$ = get_bfnd(fi,ACC_ROUTINE_DIR,SMNULL,$3,LLNULL,LLNULL);} 
           ;

opt_targets_clause: needkeyword keywordoff
	            { $$ = LLNULL; }
                  | needkeyword targets_clause
                    { $$ = $2;}                   
                  ;

spf_directive: spf_analysis
             | spf_parallel      
             | spf_transform
             | spf_parallel_reg
             | spf_end_parallel_reg
             | spf_checkpoint
	     ;

spf_analysis: SPF_ANALYSIS in_unit LEFTPAR analysis_spec_list RIGHTPAR 
             {  $$ = get_bfnd(fi,SPF_ANALYSIS_DIR,SMNULL,$4,LLNULL,LLNULL);}
             ;

spf_parallel: SPF_PARALLEL in_unit LEFTPAR parallel_spec_list RIGHTPAR 
             {  $$ = get_bfnd(fi,SPF_PARALLEL_DIR,SMNULL,$4,LLNULL,LLNULL);}
             ;

spf_transform: SPF_TRANSFORM in_unit  LEFTPAR transform_spec_list RIGHTPAR 
             {  $$ = get_bfnd(fi,SPF_TRANSFORM_DIR,SMNULL,$4,LLNULL,LLNULL);}
             ;

spf_parallel_reg: SPF_PARALLEL_REG in_unit region_name 
                  { $$ = get_bfnd(fi,SPF_PARALLEL_REG_DIR,$3,LLNULL,LLNULL,LLNULL);}
                | SPF_PARALLEL_REG in_unit region_name COMMA needkeyword  SPF_APPLY_REGION  LEFTPAR characteristic_list RIGHTPAR opt_clause_apply_fragment
                  { $$ = get_bfnd(fi,SPF_PARALLEL_REG_DIR,$3,$8,$10,LLNULL);}
                | SPF_PARALLEL_REG in_unit region_name COMMA needkeyword SPF_APPLY_FRAGMENT LEFTPAR characteristic_list RIGHTPAR opt_clause_apply_region
                  { $$ = get_bfnd(fi,SPF_PARALLEL_REG_DIR,$3,$10,$8,LLNULL);}
                ;

characteristic_list:  characteristic
	       { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
             |  characteristic_list COMMA characteristic
	       { $$ = set_ll_list($1,$3,EXPR_LIST); }	
             ;

characteristic: needkeyword SPF_CODE_COVERAGE
               { $$ = make_llnd(fi,SPF_CODE_COVERAGE_OP,LLNULL,LLNULL,SMNULL);}
              ;

opt_clause_apply_fragment: 
                           { $$ = LLNULL;}
                         | COMMA needkeyword SPF_APPLY_FRAGMENT LEFTPAR characteristic_list RIGHTPAR
                           { $$ = $5;}
                         ;

opt_clause_apply_region: 
                           { $$ = LLNULL;}
                         | COMMA needkeyword SPF_APPLY_REGION LEFTPAR characteristic_list RIGHTPAR
                           { $$ = $5;}
                         ;

spf_end_parallel_reg: SPF_END_PARALLEL_REG  in_unit
                { $$ = get_bfnd(fi,SPF_END_PARALLEL_REG_DIR,SMNULL,LLNULL,LLNULL,LLNULL);}
                ;

analysis_spec_list:  analysis_spec
	       { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
             | analysis_spec_list COMMA analysis_spec
	       { $$ = set_ll_list($1,$3,EXPR_LIST); }	
             ;

analysis_spec: analysis_reduction_spec
             | analysis_private_spec
             | analysis_process_private_spec
             | analysis_parameter_spec
             | analysis_cover_spec     
             ;

analysis_reduction_spec: needkeyword REDUCTION LEFTPAR reduction_list RIGHTPAR
               { $$ = make_llnd(fi,REDUCTION_OP,$4,LLNULL,SMNULL); }    
                     ;

analysis_private_spec:   needkeyword PRIVATE LEFTPAR variable_list RIGHTPAR
               { $$ = make_llnd(fi,ACC_PRIVATE_OP,$4,LLNULL,SMNULL);} 
                     ;

analysis_process_private_spec: needkeyword SPF_PROCESS_PRIVATE LEFTPAR variable_list RIGHTPAR
               { $$ = make_llnd(fi,SPF_PROCESS_PRIVATE_OP,$4,LLNULL,SMNULL);} 
                     ;

analysis_cover_spec: needkeyword SPF_COVER LEFTPAR integer_constant RIGHTPAR
               { $$ = make_llnd(fi,SPF_COVER_OP,$4,LLNULL,SMNULL);} 
                     ;

analysis_parameter_spec: needkeyword PARAMETER LEFTPAR spf_parameter_list RIGHTPAR
               { $$ = make_llnd(fi,SPF_PARAMETER_OP,$4,LLNULL,SMNULL);}
                     ;   
spf_parameter_list: spf_parameter
	       { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	     | spf_parameter_list COMMA spf_parameter
               { $$ = set_ll_list($1, $3, EXPR_LIST); }
             ;

spf_parameter: array_element EQUAL expr
	     { $$ = make_llnd(fi, ASSGN_OP, $1, $3, SMNULL); }
	     ;

parallel_spec_list:  parallel_spec
	       { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
             |  parallel_spec_list COMMA parallel_spec
	       { $$ = set_ll_list($1,$3,EXPR_LIST); }	
             ;

parallel_spec: parallel_shadow_spec
             | parallel_across_spec
             | parallel_remote_access_spec     
             ;

parallel_shadow_spec:  needkeyword SHADOW LEFTPAR shadow_list RIGHTPAR
                { $$ = make_llnd(fi,SHADOW_OP,$4,LLNULL,SMNULL);}
                    ;

parallel_across_spec:  needkeyword ACROSS LEFTPAR shadow_list RIGHTPAR
                { $$ = make_llnd(fi,ACROSS_OP,$4,LLNULL,SMNULL);}
                    ;

parallel_remote_access_spec: needkeyword REMOTE_ACCESS_SPEC LEFTPAR remote_data_list RIGHTPAR
                { $$ = make_llnd(fi,REMOTE_ACCESS_OP,$4,LLNULL,SMNULL);}
                           ;

transform_spec_list:  transform_spec
	       { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
             |  transform_spec_list COMMA transform_spec
	       { $$ = set_ll_list($1,$3,EXPR_LIST); }	
             ;

transform_spec: needkeyword SPF_NOINLINE
                { $$ = make_llnd(fi,SPF_NOINLINE_OP,LLNULL,LLNULL,SMNULL);}
              | needkeyword SPF_FISSION LEFTPAR ident_list RIGHTPAR
                { $$ = make_llnd(fi,SPF_FISSION_OP,$4,LLNULL,SMNULL);}
              | needkeyword SPF_EXPAND 
                { $$ = make_llnd(fi,SPF_EXPAND_OP,LLNULL,LLNULL,SMNULL);}
              | needkeyword SPF_EXPAND LEFTPAR ident_list RIGHTPAR
                { $$ = make_llnd(fi,SPF_EXPAND_OP,$4,LLNULL,SMNULL);}
           /*   | needkeyword SPF_SHRINK LEFTPAR ident LEFTPAR digit_list RIGHTPAR RIGHTPAR  */
              | needkeyword SPF_SHRINK LEFTPAR array_element_list RIGHTPAR
                { $$ = make_llnd(fi,SPF_SHRINK_OP,$4,LLNULL,SMNULL);}
              | needkeyword SPF_UNROLL 
                { $$ = make_llnd(fi,SPF_UNROLL_OP,LLNULL,LLNULL,SMNULL);}
              | needkeyword SPF_UNROLL LEFTPAR unroll_list RIGHTPAR
                { $$ = make_llnd(fi,SPF_UNROLL_OP,$4,LLNULL,SMNULL);}
              | needkeyword SPF_MERGE 
                { $$ = make_llnd(fi,SPF_MERGE_OP,LLNULL,LLNULL,SMNULL);}
              ;

unroll_list: expr COMMA expr COMMA expr
             {
               $$ = set_ll_list($1, $3, EXPR_LIST);
               $$ = set_ll_list($$, $5, EXPR_LIST);
             } 
           ;

region_name: name
           { $$ = make_parallel_region($1);}
           ; 
  
array_element_list: array_element
	           { $$ = set_ll_list($1, LLNULL, EXPR_LIST); }
	          | array_element_list COMMA array_element 
	           { $$ = set_ll_list($1, $3, EXPR_LIST); }
	          ;

spf_checkpoint:  SPF_CHECKPOINT in_unit LEFTPAR checkpoint_spec_list RIGHTPAR 
             {  $$ = get_bfnd(fi,SPF_CHECKPOINT_DIR,SMNULL,$4,LLNULL,LLNULL);}
             ;

checkpoint_spec_list:  checkpoint_spec
	       { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
             |  checkpoint_spec_list COMMA checkpoint_spec
	       { $$ = set_ll_list($1,$3,EXPR_LIST); }	
             ;

checkpoint_spec: needkeyword TYPE LEFTPAR spf_type_list RIGHTPAR
                { $$ = make_llnd(fi,SPF_TYPE_OP,$4,LLNULL,SMNULL);}
              | needkeyword VARLIST LEFTPAR ident_list RIGHTPAR
                { $$ = make_llnd(fi,SPF_VARLIST_OP,$4,LLNULL,SMNULL);}
              | needkeyword SPF_EXCEPT LEFTPAR ident_list RIGHTPAR
                { $$ = make_llnd(fi,SPF_EXCEPT_OP,$4,LLNULL,SMNULL);}
              | needkeyword SPF_FILES_COUNT LEFTPAR expr RIGHTPAR
                { $$ = make_llnd(fi,SPF_FILES_COUNT_OP,$4,LLNULL,SMNULL);}
              | needkeyword SPF_INTERVAL LEFTPAR interval_spec COMMA expr RIGHTPAR
                { $$ = make_llnd(fi,SPF_INTERVAL_OP,$4,$6,SMNULL);}
              ;

spf_type_list: spf_type
	       { $$ = set_ll_list($1,LLNULL,EXPR_LIST); }
             | spf_type_list COMMA spf_type
	       { $$ = set_ll_list($1,$3,EXPR_LIST); }	
             ;

spf_type:       needkeyword ACC_ASYNC
              { $$ = make_llnd(fi,ACC_ASYNC_OP, LLNULL,LLNULL,SMNULL);}
             |  needkeyword SPF_FLEXIBLE
              { $$ = make_llnd(fi,SPF_FLEXIBLE_OP, LLNULL,LLNULL,SMNULL);}
             ;

interval_spec:  needkeyword SPF_TIME
                { $$ = make_llnd(fi,SPF_TIME_OP, LLNULL,LLNULL,SMNULL);}
             |  needkeyword SPF_ITER
                { $$ = make_llnd(fi,SPF_ITER_OP, LLNULL,LLNULL,SMNULL);}
             ;

in_unit: 
             { if(position==IN_OUTSIDE)
                 err("Misplaced SPF-directive",103);
             }
             ;