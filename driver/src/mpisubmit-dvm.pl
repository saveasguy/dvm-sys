#!/usr/bin/perl -I /etc/LoadL

###############################################################
#    mpisubmit.pl -- script for submitting simple mpi jobs    #
#                    to LSF via ' llsubmit - '                #
###############################################################

use strict;
use Getopt::Long;
use POSIX;

use constant NODES_ALL => 4;
use constant CPUS_PER_NODE => 2;
use constant CPUS_ALL => CPUS_PER_NODE * NODES_ALL;
use constant CORES_PER_CPU => 10;
use constant CORES_PER_NODE => CORES_PER_CPU * CPUS_PER_NODE;
use constant CORES_ALL => CORES_PER_NODE * NODES_ALL;
use constant THREADS_PER_CORE => 8;
use constant THREADS_PER_CPU => THREADS_PER_CORE * CORES_PER_NODE;
use constant THREADS_PER_NODE => THREADS_PER_CPU * CPUS_PER_NODE;
use constant THREADS_ALL => THREADS_PER_NODE * NODES_ALL;

#use CmcSubmitUtils;

my $submit_file_name = "| bsub "; # file name or filter to stream commands out

my $user = `id -nu`;
chomp ($user);

# !!! use 'threads' as DVMH_PPN !!!
# !!! use 'gpu' as DVMH_EXCLUSIVE !!!

# job file parameters
my %cmd =(
		'processes' => '1', # nubmer of MPI processes
        'threads' => '1', # number of OpenMP threads per MPI process
        'cores' => '1', # number of cores
		'wtime' => '00:15', #wall time limit
        'gpu' => 0,
		'in' => '', # file stdin flows from
		'out' => '%J.out', # file stdout flows to
		'err' => '%J.err', # file stderr flows to

        'debug' => 0, # dump to stdout

		'efile' => '', #file to execute
		'args' => '', #its argument
	);				


# parse command line
sub printHelp;
my $was_help = 0;

GetOptions (
		'processes=i' => \$cmd{"processes"},
                'nprocs=i' => sub {
		    die "error: Option --nproc (-n) is obsolated. Please, use --processes (-p)\n";
                },
        'threads=i' => \$cmd{"threads"},
		'wtime=s' => \$cmd{"wtime"},
        'gpu:i' => sub {$cmd{"gpu"} = @_[1] || 1;},
		'stdin=s' => \$cmd{"in"},
		'stdout=s' => \$cmd{"out"},
		'stderr=s' => \$cmd{"err"},
                'debug' => \$cmd{"debug"},
		'help' => sub {$was_help = 1;},
	);

$was_help && printHelp;
#$was_class && printClass;

my @tags;
($cmd{"efile"}, @tags) = @ARGV;
$cmd{"args"} =  join ( ' ', @tags); 
 
( $cmd{"efile"} ) || ( warn ("error: you should specify executable\n") && printHelp);

# name stout and stderr files
my $efile_name = (split /\//, $cmd{"efile"} ) [-1];
$cmd{"out"} = "$efile_name." . $cmd{"out"}
	if ( $cmd{"out"} eq '%J.out' );
$cmd{"err"} = "$efile_name." . $cmd{"err"}
	if ( $cmd{"err"} eq '%J.err' );


$cmd{"cores"} = $cmd{"processes"};

$cmd{ "cores" } <= CORES_ALL && $cmd{ "cores" } >= 1 ||
	die "error: you can't allocate such number of nodes\n";


if ( $cmd{'debug'} ) {
    $submit_file_name = "| cat ";
}

open (SUBMIT_STREAM, $submit_file_name);
select SUBMIT_STREAM;

print "# this file was automaticly created by mpisubmit.pl script for $user #\n";

print "#BSUB -x\n" if $cmd{"gpu"};
print "#BSUB -n ", $cmd{"cores"}, "\n";
print "#BSUB -R span[ptile=" . $cmd{"threads"} . "]\n";    
print "#BSUB -W ", $cmd{"wtime"}, "\n";
print "#BSUB -gpu \"num=2:mode=shared:mps=yes\"\n";
print "#BSUB -i ", $cmd{"in"}, "\n" if $cmd{'in'};
print "#BSUB -o ", $cmd{"out"}, "\n";
print "#BSUB -e ", $cmd{"err"}, "\n";

print " mpiexec";
print " " . $cmd{'efile'};
print " " . $cmd{'args'} if $cmd{'args'} ne '';;
print "\n";

close (SUBMIT_STREAM);


# here's the usage information printing sub
sub printHelp
{
    warn ("usage: mpisubmit.pl {<option_value_pair>} <executable to submit> -- <args>\n");
    warn (" where <option_value_pair> could be:\n");
    warn ("\t( -p | --processes ) <number of MPI processes>,\n\t\tdefault is 1\n");
    warn ("\t( -t | --threads ) <number of OpenMP threads per MPI process>,\n\t\tdefault is 1\n");
#    warn ("\t( -m | --mode ) <mode to be used: smp | dual | vn>,\n\t\tdefault is smp\n");
    warn ("\t( -w | --wtime ) <wall clock limit>,\n\t\tdefault is 00:15\n");
    warn ("\t( -g | --gpu ) <number of GPUs per host to use>,\n\t\tdefault is none\n");
#    warn ("\t( -e | --env ) <environment to be passed >\n\t\t\"env=val env=val ...\"\n");
#    warn ("\t( -t | --top ) <topology to be used: TORUS | MESH | PREFER_TORUS>,\n\t\tdefault is PREFER_TORUS\"\n");

    warn ("\t --stdout <file to direct stdout to>,\n\t\tdefault is '\$(jobid).out'\n");
    warn ("\t --stderr <file to direct sterr to>,\n\t\tdefault is '\$(jobid).err'\n");
    warn ("\t --stdin <file to direct stdin from>,\n\t\tno default\n");

    warn ("\t( -d | --debug ) prints the jcf file to stdout instead of submiting to LSF\n");

#    warn ("\t( -c | --class ) prints the name of class to be used for the specified wtime and nproc\n");
    warn ("\t( -h | --help ) prints this message out\n");

    warn ("\ni.e. mpisubmit.pl -w 00:15 -p 32 a.out\n");
    die ("or mpisubmit.pl -w 00:15 -p 32 a.out -- 0.01\n");
}

