      program jac2d
      parameter (l = 8000,itmax = 100)
      real  eps,maxeps
      double precision  startt,endt,dvtime

! DVMH declarations 
      external dvmh_line,dvmh_scope_end,dvmh_scope_start,dvmh_finish,dvm
     &h_init2,ftcntr,lexit,clfdvm
      integer*8  dvmh_string,arrcpy,getach,getad,getaf,getal,getai,align
     &,crtda,distr,crtamv
      integer*8  dvm000(23)
      integer*8  a(38),b(38)
      character*9 ::  filenm001='arrcpy.f'//char (0)
      integer*8  dvm0c9,dvm0c8,dvm0c7,dvm0c6,dvm0c5,dvm0c4,dvm0c3,dvm0c2
     &,dvm0c1,dvm0c0
      parameter (dvm0c9 = 9,dvm0c8 = 8,dvm0c7 = 7,dvm0c6 = 6,dvm0c5 = 5,
     &dvm0c4 = 4,dvm0c3 = 3,dvm0c2 = 2,dvm0c1 = 1,dvm0c0 = 0)
      character  ch000m(0:64)
      logical  l0000m(0:64)
      double precision  d0000m(0:64)
      real  r0000m(0:64)
      integer  i0000m(0:64)
      equivalence (l0000m,d0000m,r0000m,i0000m)
      common /mem000/i0000m
      save 
      call dvmh_line(9_8,dvmh_string (filenm001))
      dvm000(1) = dvm0c6
      dvm000(2) = getai (dvm000(1))
      dvm000(3) = getai (i0000m(0))
      dvm000(4) = getal (l0000m(0))
      dvm000(5) = getaf (r0000m(0))
      dvm000(6) = getad (d0000m(0))
      dvm000(7) = getach (ch000m(0))
      dvm000(8) = getai (dvm000(2))
      dvm000(9) = getai (i0000m(1))
      dvm000(10) = getal (l0000m(1))
      dvm000(11) = getaf (r0000m(1))
      dvm000(12) = getad (d0000m(1))
      dvm000(13) = getach (ch000m(1))
      i0000m(0) = 8
      i0000m(1) = 4
      i0000m(2) = 4
      i0000m(3) = 4
      i0000m(4) = 8
      i0000m(5) = 1
      i0000m(10) = 2
      i0000m(11) = 1
      i0000m(12) = 1
      i0000m(13) = 3
      i0000m(14) = 4
      i0000m(15) = 5
      call ftcntr(6,dvm000(2),dvm000(8),i0000m(0),i0000m(10))
      dvm000(1) = 1

!$    dvm000(1) = dvm000(1) + 8 
      call dvmh_init2(dvm000(1))
      call dvmh_scope_start()
      dvm000(4) = 2
      dvm000(5) = 1
      dvm000(6) = 0
      dvm000(7) = 0
      call dvmh_line(6_8,dvmh_string (filenm001))
      dvm000(8) = 8000
      dvm000(9) = 8000
      dvm000(10) = crtamv (dvm0c0,dvm0c2,dvm000(8),dvm0c0)
      dvm000(11) = distr (dvm000(10),dvm0c0,dvm0c2,dvm000(4),dvm000(6))
      dvm000(12) = 1
      dvm000(13) = 1
      a(2:3) = 1
      a(5) = 1
      a(6) = 1
      a(7) = 8
      dvm000(14) = crtda (a(1),dvm0c1,i0000m,dvm0c2,dvm0c4,dvm000(8),dvm
     &0c0,dvm0c0,dvm000(12),dvm000(12))
      dvm000(15) = 1
      dvm000(16) = 2
      dvm000(17) = 1
      dvm000(18) = 1
      dvm000(19) = 0
      dvm000(20) = 0
      dvm000(21) = align (a(1),dvm000(10),dvm000(15),dvm000(17),dvm000(1
     &9))
      dvm000(12) = 1
      dvm000(13) = 2
      dvm000(14) = 1
      dvm000(15) = 1
      dvm000(16) = 0
      dvm000(17) = 0
      call dvmh_line(7_8,dvmh_string (filenm001))
      dvm000(18) = 8000
      dvm000(19) = 8000
      dvm000(20) = 1
      dvm000(21) = 1
      b(2:3) = 1
      b(5) = 1
      b(6) = 1
      b(7) = 8
      dvm000(22) = crtda (b(1),dvm0c1,i0000m,dvm0c2,dvm0c4,dvm000(18),dv
     &m0c0,dvm0c0,dvm000(20),dvm000(20))
      dvm000(23) = align (b(1),a(1),dvm000(12),dvm000(14),dvm000(16))
      call dvmh_line(9_8,dvmh_string (filenm001))

!        arrays A and B  with block distribution 
      dvm000(11) = (-1)
      dvm000(12) = (-1)
      dvm000(17) = (-1)
      dvm000(18) = (-1)
      dvm000(23) = arrcpy (b(1),dvm000(17),dvm000(19),dvm000(21),a(1),dv
     &m000(11),dvm000(13),dvm000(15),dvm0c0)
      call dvmh_line(10_8,dvmh_string (filenm001))
      call dvmh_scope_end()
      call dvmh_line(10_8,dvmh_string (filenm001))
      call clfdvm()
      call dvmh_finish()
      call lexit(dvm0c0)
      end


!-----------------------------------------------------------------------

