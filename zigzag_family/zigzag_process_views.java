{
    // zigzag_process_views.java — zigzag 簇退蚀过程温度场/等效热应力场显示脚本
    //
    // 基于 zigzag_baseline.java 新建：
    //   1) 保留原 zigzag S2S + 电压搜索 + 退蚀寿命流程；
    //   2) 记录退蚀过程历史，结束后重建 4 个独立阶段 dataset；
    //   3) 每个阶段默认显示温度场 T，可在 COMSOL 中把表达式切换为 solid.mises。
    //      Solid Mechanics 使用热膨胀载荷求解 von Mises 应力场。
    //      注意：该脚本需要 COMSOL 支持 Solid Mechanics/结构力学接口。
    //
    // 原始说明：
    // zigzag_baseline.java — 完整生命周期仿真
    // 赛题：MultipleSpectralBands S2S + 电压搜索 + 侵蚀循环
    // 修复：solver 只 clearSolution()，不 remove()，避免 result dataset 失效
    // 用法: COMSOL Desktop → Java Shell → Ctrl+Enter

    // ================================================================
    //  常量
    // ================================================================
    int    N_RUNS            = 8;
    double L_RUN             = 104e-3;
    double r0Value           = 2.5e-3;
    double L0Value           = 15e-3;
    double terminalStubLen   = 0.5e-3;
    double V0                = Math.PI * r0Value * r0Value * L0Value;
    double rhoMassWValue     = 19350.0;
    double tempLimitK        = 3273.15;
    double outerSphereMargin = 1.05;
    double voltageUpper      = 100.0;
    double voltageFloor      = 1e-3;
    double volTol            = 0.03;
    double currentTol        = 1e-9;
    double voltageTol        = 0.05;
    int    maxVoltageIters   = 20;
    double AevVal            = 3.9e9;
    double BevVal            = 1.023e5;
    double failureFraction   = 0.20;
    int    maxErosionSteps   = 50;
    int    MAX_BLOCK_SLOTS   = 64;

    // ================================================================
    //  材料表达式
    // ================================================================
    String rhoeExpr = "max(1e-10[ohm*m], 5.5e-8[ohm*m]*(1+0.003836*(T-293.15[K])/1[K]"
        + "+7.55e-7*((T-293.15[K])/1[K])^2))";
    String kExpr    = "max(75[W/(m*K)],175[W/(m*K)]-0.032[W/(m*K^2)]*(T-293.15[K]))";
    String cpExpr   = "min(195[J/(kg*K)],132[J/(kg*K)]+0.020[J/(kg*K^2)]*(T-293.15[K]))";

    // ================================================================
    //  Planck f03 六项级数
    // ================================================================
    String x03T = "(c2bb/(lam03*T))";
    String serT = "";
    for (int n = 1; n <= 6; n++) {
        int n2=n*n, n3=n2*n, n4=n3*n;
        String tT  = "exp(-"+n+"*"+x03T+")*("+x03T+"^3/"+n+"+3*"+x03T+"^2/"+n2+"+6*"+x03T+"/"+n3+"+6/"+n4+")";
        serT = (n==1) ? tT : serT+"+"+tT;
    }
    String f03T = "min(1,max(0,(15/pi^4)*("+serT+")))";
    String q03Expr  = "eps03*sigmaSB*(("+f03T+")*T^4)";
    String qradExpr = "sigmaSB*(epsRest*T^4+(eps03-epsRest)*(("+f03T+")*T^4))";

    // ================================================================
    //  路径 & 几何计算
    // ================================================================
    double zFirst = 0.8e-3;
    double zLast  = L0Value - zFirst;
    double zStep  = (zLast - zFirst) / (N_RUNS - 1);
    double[] xTargets = new double[N_RUNS];
    double[] zLevels  = new double[N_RUNS];
    for (int i = 0; i < N_RUNS; i++) {
        xTargets[i] = ((i%2)==0) ? L_RUN : 0.0;
        zLevels[i]  = zFirst + i*zStep;
    }
    int np = 2 + 2*N_RUNS;
    double[][] pts = new double[np][2];
    int ki = 0; double cx = 0.0;
    pts[ki][0]=0.0; pts[ki][1]=terminalStubLen; ki++;
    for (int i=0; i<N_RUNS; i++) {
        pts[ki][0]=cx; pts[ki][1]=zLevels[i]; ki++;
        cx=xTargets[i];
        pts[ki][0]=cx; pts[ki][1]=zLevels[i]; ki++;
    }
    pts[ki][0]=cx; pts[ki][1]=L0Value-terminalStubLen;

    double plen=0.0;
    for (int i=0; i<np-1; i++)
        plen += Math.abs(pts[i+1][0]-pts[i][0]) + Math.abs(pts[i+1][1]-pts[i][1]);
    double side0 = Math.sqrt((V0 - 2.0*Math.PI*r0Value*r0Value*terminalStubLen) / plen);
    double halfS = 0.5*side0;
    System.out.println("PATH_LENGTH_MM="+(plen*1e3));
    System.out.println("SIDE_MM="+(side0*1e3));

    // ================================================================
    //  初始 block 数据
    // ================================================================
    double[]  blkX0       = new double[MAX_BLOCK_SLOTS];
    double[]  blkY0       = new double[MAX_BLOCK_SLOTS];
    double[]  blkZ0       = new double[MAX_BLOCK_SLOTS];
    double[]  blkSX       = new double[MAX_BLOCK_SLOTS];
    double[]  blkSY       = new double[MAX_BLOCK_SLOTS];
    double[]  blkSZ       = new double[MAX_BLOCK_SLOTS];
    boolean[] blkIsHoriz  = new boolean[MAX_BLOCK_SLOTS];
    int blkCount = 0;

    for (int i=0; i<np-1; i++) {
        double dx=pts[i+1][0]-pts[i][0], dz=pts[i+1][1]-pts[i][1];
        if (Math.abs(dx)<1e-12 && Math.abs(dz)<1e-12) continue;
        double extS=(i==0)?0.0:halfS, extE=(i==np-2)?0.0:halfS;
        if (Math.abs(dx)>1e-12) {
            double d=(dx>0)?1.0:-1.0;
            double xa=pts[i][0]-d*extS, xb=pts[i+1][0]+d*extE;
            blkX0[blkCount]=Math.min(xa,xb); blkY0[blkCount]=-halfS; blkZ0[blkCount]=pts[i][1]-halfS;
            blkSX[blkCount]=Math.abs(xb-xa); blkSY[blkCount]=side0; blkSZ[blkCount]=side0;
            blkIsHoriz[blkCount]=true;
        } else {
            double d=(dz>0)?1.0:-1.0;
            double za=pts[i][1]-d*extS, zb=pts[i+1][1]+d*extE;
            blkX0[blkCount]=pts[i][0]-halfS; blkY0[blkCount]=-halfS; blkZ0[blkCount]=Math.min(za,zb);
            blkSX[blkCount]=side0; blkSY[blkCount]=side0; blkSZ[blkCount]=Math.abs(zb-za);
            blkIsHoriz[blkCount]=false;
        }
        blkCount++;
    }
    System.out.println("BLOCK_COUNT="+blkCount);

    // ================================================================
    //  外接球半径
    // ================================================================
    double Renv0;
    {
        double md=0.0, zc=0.5*L0Value;
        for (int bi=0; bi<blkCount; bi++) {
            for (double xx:new double[]{blkX0[bi],blkX0[bi]+blkSX[bi]})
                for (double yy:new double[]{blkY0[bi],blkY0[bi]+blkSY[bi]})
                    for (double zz:new double[]{blkZ0[bi],blkZ0[bi]+blkSZ[bi]}) {
                        double d=Math.sqrt(xx*xx+yy*yy+(zz-zc)*(zz-zc));
                        if (d>md) md=d;
                    }
        }
        for (double rx:new double[]{-r0Value,r0Value})
            for (double ry:new double[]{-r0Value,r0Value})
                for (double zz:new double[]{0,terminalStubLen,L0Value-terminalStubLen,L0Value}) {
                    double d=Math.sqrt(rx*rx+ry*ry+(zz-zc)*(zz-zc));
                    if (d>md) md=d;
                }
        Renv0=outerSphereMargin*md;
    }
    System.out.println("R_ENV_MM="+(Renv0*1e3));
    final int stageBlkCount = blkCount;

    class StageDatasetOps {
        void rebuildStageGeometry(double geomSide, double shrink, double renv) {
            try{model.component("comp1").geom("geom1").feature().remove("uniZZ");}catch(Exception e){}
            try{model.component("comp1").geom("geom1").feature().remove("term_in");}catch(Exception e){}
            try{model.component("comp1").geom("geom1").feature().remove("term_out");}catch(Exception e){}
            for(int bi=0;bi<MAX_BLOCK_SLOTS;bi++) try{model.component("comp1").geom("geom1").feature().remove("blk_"+(bi+1));}catch(Exception e){}

            java.util.ArrayList<String> tl=new java.util.ArrayList<String>();
            for(int bi=0;bi<stageBlkCount;bi++){
                String tag="blk_"+(bi+1); tl.add(tag);
                double nx0,ny0,nz0,nsx,nsy,nsz;
                if(blkIsHoriz[bi]){nx0=blkX0[bi];ny0=blkY0[bi]+shrink;nz0=blkZ0[bi]+shrink;nsx=blkSX[bi];nsy=geomSide;nsz=geomSide;}
                else{nx0=blkX0[bi]+shrink;ny0=blkY0[bi]+shrink;nz0=blkZ0[bi];nsx=geomSide;nsy=geomSide;nsz=blkSZ[bi];}
                model.component("comp1").geom("geom1").create(tag,"Block");
                model.component("comp1").geom("geom1").feature(tag).set("size",new String[]{Double.toString(nsx)+"[m]",Double.toString(nsy)+"[m]",Double.toString(nsz)+"[m]"});
                model.component("comp1").geom("geom1").feature(tag).set("pos",new String[]{Double.toString(nx0)+"[m]",Double.toString(ny0)+"[m]",Double.toString(nz0)+"[m]"});
            }
            model.component("comp1").geom("geom1").create("term_in","Cylinder");
            model.component("comp1").geom("geom1").feature("term_in").set("r",Double.toString(r0Value)+"[m]");
            model.component("comp1").geom("geom1").feature("term_in").set("h",Double.toString(terminalStubLen)+"[m]");
            model.component("comp1").geom("geom1").feature("term_in").set("pos",new String[]{"0[m]","0[m]","0[m]"});
            model.component("comp1").geom("geom1").create("term_out","Cylinder");
            model.component("comp1").geom("geom1").feature("term_out").set("r",Double.toString(r0Value)+"[m]");
            model.component("comp1").geom("geom1").feature("term_out").set("h",Double.toString(terminalStubLen)+"[m]");
            model.component("comp1").geom("geom1").feature("term_out").set("pos",new String[]{"0[m]","0[m]",Double.toString(L0Value-terminalStubLen)+"[m]"});
            String[] ut=new String[stageBlkCount+2];
            for(int bi=0;bi<stageBlkCount;bi++) ut[bi]=tl.get(bi);
            ut[stageBlkCount]="term_in"; ut[stageBlkCount+1]="term_out";
            model.component("comp1").geom("geom1").create("uniZZ","Union");
            model.component("comp1").geom("geom1").feature("uniZZ").selection("input").set(ut);
            model.component("comp1").geom("geom1").feature("uniZZ").set("intbnd",false);
            model.component("comp1").geom("geom1").run();
            model.param().set("RenvZZ",Double.toString(renv)+"[m]");
            model.param().set("AenvZZ",Double.toString(4*Math.PI*renv*renv)+"[m^2]");
            model.component("comp1").mesh("mesh1").feature("size").set("hauto",5);
            try{model.component("comp1").mesh("mesh1").feature("ftet1");}
            catch(Exception e){model.component("comp1").mesh("mesh1").create("ftet1","FreeTet");}
            model.component("comp1").mesh("mesh1").run();
        }
    }

    // ================================================================
    //  模型骨架
    // ================================================================
    model.label("zigzag_process_views.mph");
    model.param().set("sigmaSB","5.670374419e-8[W/(m^2*K^4)]");
    model.param().set("eps03","0.35");
    model.param().set("epsRest","0.15");
    model.param().set("rhoMassW","19350[kg/m^3]");
    model.param().set("Tamb","293.15[K]");
    model.param().set("Telectrode","293.15[K]");
    model.param().set("EW","411[GPa]");
    model.param().set("nuW","0.28");
    model.param().set("alphaW","4.5e-6[1/K]");
    model.param().set("Vapp","1[V]");
    model.param().set("lam03","3[um]");
    model.param().set("c2bb","1.438776877e-2[m*K]");
    model.param().set("r0",Double.toString(r0Value)+"[m]");
    model.param().set("L0","15[mm]");
    model.param().set("Aev","3.9e9[kg/(m^2*s)]");
    model.param().set("Bev","1.023e5[K]");
    model.param().set("RenvZZ",Double.toString(Renv0)+"[m]");
    model.param().set("AenvZZ",Double.toString(4*Math.PI*Renv0*Renv0)+"[m^2]");

    model.component().create("comp1");
    model.component("comp1").geom().create("geom1",3);
    model.component("comp1").geom("geom1").lengthUnit("mm");
    model.component("comp1").physics().create("ec","ConductiveMedia","geom1");
    model.component("comp1").physics().create("ht","HeatTransfer","geom1");
    model.component("comp1").physics().create("solid","SolidMechanics","geom1");
    try{model.component("comp1").physics("ht").feature("init1").set("Tinit","1500[K]");}catch(Exception e){}
    model.component("comp1").material().create("mat1","Common");
    model.component("comp1").material("mat1").label("Tungsten");
    model.component("comp1").material("mat1").selection().all();
    model.component("comp1").mesh().create("mesh1","geom1");
    model.study().create("std1");
    model.study("std1").create("stat","Stationary");

    model.multiphysics().create("emh1","ElectromagneticHeatSource","geom1",3);
    model.multiphysics("emh1").selection().all();
    model.multiphysics("emh1").set("EMHeat_physics","ec");
    model.multiphysics("emh1").set("Heat_physics","ht");

    model.component("comp1").material("mat1").propertyGroup("def").set("density",new String[]{"rhoMassW"});
    model.component("comp1").material("mat1").propertyGroup("def").set("electricconductivity",new String[]{"1/("+rhoeExpr+")"});
    model.component("comp1").material("mat1").propertyGroup("def").set("thermalconductivity",new String[]{kExpr});
    model.component("comp1").material("mat1").propertyGroup("def").set("heatcapacity",new String[]{cpExpr});
    model.component("comp1").material("mat1").propertyGroup("def").set("youngsmodulus",new String[]{"EW"});
    model.component("comp1").material("mat1").propertyGroup("def").set("poissonsratio",new String[]{"nuW"});
    model.component("comp1").material("mat1").propertyGroup("def").set("thermalexpansioncoefficient",new String[]{"alphaW"});

    // ================================================================
    //  辅助：重建几何 + 网格（内联复用，geomSide/shrink 由外部设置）
    //  ★ 不重建物理场，避免 result dataset 引用失效 ★
    // ================================================================

    // --- 初始几何 + 物理（geomSide = side0, shrink = 0）---
    {
        double gs=side0, sh=0.0;
        try{model.component("comp1").geom("geom1").feature().remove("uniZZ");}catch(Exception e){}
        try{model.component("comp1").geom("geom1").feature().remove("term_in");}catch(Exception e){}
        try{model.component("comp1").geom("geom1").feature().remove("term_out");}catch(Exception e){}
        for(int bi=0;bi<MAX_BLOCK_SLOTS;bi++) try{model.component("comp1").geom("geom1").feature().remove("blk_"+(bi+1));}catch(Exception e){}

        java.util.ArrayList<String> tl=new java.util.ArrayList<String>();
        for(int bi=0;bi<blkCount;bi++){
            String tag="blk_"+(bi+1); tl.add(tag);
            double nx0,ny0,nz0,nsx,nsy,nsz;
            if(blkIsHoriz[bi]){nx0=blkX0[bi];ny0=blkY0[bi]+sh;nz0=blkZ0[bi]+sh;nsx=blkSX[bi];nsy=gs;nsz=gs;}
            else{nx0=blkX0[bi]+sh;ny0=blkY0[bi]+sh;nz0=blkZ0[bi];nsx=gs;nsy=gs;nsz=blkSZ[bi];}
            model.component("comp1").geom("geom1").create(tag,"Block");
            model.component("comp1").geom("geom1").feature(tag).set("size",new String[]{Double.toString(nsx)+"[m]",Double.toString(nsy)+"[m]",Double.toString(nsz)+"[m]"});
            model.component("comp1").geom("geom1").feature(tag).set("pos",new String[]{Double.toString(nx0)+"[m]",Double.toString(ny0)+"[m]",Double.toString(nz0)+"[m]"});
        }
        model.component("comp1").geom("geom1").create("term_in","Cylinder");
        model.component("comp1").geom("geom1").feature("term_in").set("r",Double.toString(r0Value)+"[m]");
        model.component("comp1").geom("geom1").feature("term_in").set("h",Double.toString(terminalStubLen)+"[m]");
        model.component("comp1").geom("geom1").feature("term_in").set("pos",new String[]{"0[m]","0[m]","0[m]"});
        model.component("comp1").geom("geom1").create("term_out","Cylinder");
        model.component("comp1").geom("geom1").feature("term_out").set("r",Double.toString(r0Value)+"[m]");
        model.component("comp1").geom("geom1").feature("term_out").set("h",Double.toString(terminalStubLen)+"[m]");
        model.component("comp1").geom("geom1").feature("term_out").set("pos",new String[]{"0[m]","0[m]",Double.toString(L0Value-terminalStubLen)+"[m]"});
        String[] ut=new String[blkCount+2];
        for(int bi=0;bi<blkCount;bi++) ut[bi]=tl.get(bi);
        ut[blkCount]="term_in"; ut[blkCount+1]="term_out";
        model.component("comp1").geom("geom1").create("uniZZ","Union");
        model.component("comp1").geom("geom1").feature("uniZZ").selection("input").set(ut);
        model.component("comp1").geom("geom1").feature("uniZZ").set("intbnd",false);
        model.component("comp1").geom("geom1").run();

        // Box selections（坐标驱动，几何重建后自动刷新）
        try{model.component("comp1").selection().remove("selInZZ");}catch(Exception e){}
        try{model.component("comp1").selection().remove("selOutZZ");}catch(Exception e){}
        try{model.component("comp1").selection().remove("selFreeZZ");}catch(Exception e){}
        for(int bi=0;bi<MAX_BLOCK_SLOTS;bi++) try{model.component("comp1").selection().remove("selBlkLat_"+(bi+1));}catch(Exception e){}
        model.component("comp1").selection().create("selInZZ","Box");
        model.component("comp1").selection("selInZZ").geom("geom1",2);
        model.component("comp1").selection("selInZZ").set("condition","inside");
        model.component("comp1").selection("selInZZ").set("xmin",-10.0); model.component("comp1").selection("selInZZ").set("xmax",10.0);
        model.component("comp1").selection("selInZZ").set("ymin",-10.0); model.component("comp1").selection("selInZZ").set("ymax",10.0);
        model.component("comp1").selection("selInZZ").set("zmin",-1e-6);  model.component("comp1").selection("selInZZ").set("zmax",1e-6);
        model.component("comp1").selection().create("selOutZZ","Box");
        model.component("comp1").selection("selOutZZ").geom("geom1",2);
        model.component("comp1").selection("selOutZZ").set("condition","inside");
        model.component("comp1").selection("selOutZZ").set("xmin",-10.0); model.component("comp1").selection("selOutZZ").set("xmax",10.0);
        model.component("comp1").selection("selOutZZ").set("ymin",-10.0); model.component("comp1").selection("selOutZZ").set("ymax",10.0);
        model.component("comp1").selection("selOutZZ").set("zmin",14.999999); model.component("comp1").selection("selOutZZ").set("zmax",15.000001);
        model.component("comp1").selection().create("selFreeZZ","Box");
        model.component("comp1").selection("selFreeZZ").geom("geom1",2);
        model.component("comp1").selection("selFreeZZ").set("condition","intersects");
        model.component("comp1").selection("selFreeZZ").set("xmin",-1.0);
        model.component("comp1").selection("selFreeZZ").set("xmax",110.0);
        model.component("comp1").selection("selFreeZZ").set("ymin",-10.0);
        model.component("comp1").selection("selFreeZZ").set("ymax",10.0);
        model.component("comp1").selection("selFreeZZ").set("zmin",1e-6);
        model.component("comp1").selection("selFreeZZ").set("zmax",14.999999);

        for(int bi=0;bi<blkCount;bi++){
            String sTag="selBlkLat_"+(bi+1);
            double nx0,ny0,nz0,nsx,nsy,nsz;
            if(blkIsHoriz[bi]){nx0=blkX0[bi];ny0=blkY0[bi]+sh;nz0=blkZ0[bi]+sh;nsx=blkSX[bi];nsy=gs;nsz=gs;}
            else{nx0=blkX0[bi]+sh;ny0=blkY0[bi]+sh;nz0=blkZ0[bi];nsx=gs;nsy=gs;nsz=blkSZ[bi];}
            double pad=Math.max(1e-5,0.01*gs*1e3);
            model.component("comp1").selection().create(sTag,"Box");
            model.component("comp1").selection(sTag).geom("geom1",2);
            model.component("comp1").selection(sTag).set("condition","intersects");
            if(blkIsHoriz[bi]){
                double x0=(nx0+0.10*nsx)*1e3, x1=(nx0+0.90*nsx)*1e3;
                if(x1<=x0){x0=nx0*1e3; x1=(nx0+nsx)*1e3;}
                model.component("comp1").selection(sTag).set("xmin",x0);
                model.component("comp1").selection(sTag).set("xmax",x1);
                model.component("comp1").selection(sTag).set("ymin",ny0*1e3-pad);
                model.component("comp1").selection(sTag).set("ymax",(ny0+nsy)*1e3+pad);
                model.component("comp1").selection(sTag).set("zmin",nz0*1e3-pad);
                model.component("comp1").selection(sTag).set("zmax",(nz0+nsz)*1e3+pad);
            } else {
                double z0=(nz0+0.10*nsz)*1e3, z1=(nz0+0.90*nsz)*1e3;
                if(z1<=z0){z0=nz0*1e3; z1=(nz0+nsz)*1e3;}
                model.component("comp1").selection(sTag).set("xmin",nx0*1e3-pad);
                model.component("comp1").selection(sTag).set("xmax",(nx0+nsx)*1e3+pad);
                model.component("comp1").selection(sTag).set("ymin",ny0*1e3-pad);
                model.component("comp1").selection(sTag).set("ymax",(ny0+nsy)*1e3+pad);
                model.component("comp1").selection(sTag).set("zmin",z0);
                model.component("comp1").selection(sTag).set("zmax",z1);
            }
        }

        // EC 边界条件
        model.component("comp1").physics("ec").create("potZZ","ElectricPotential",2);
        model.component("comp1").physics("ec").feature("potZZ").selection().named("selInZZ");
        model.component("comp1").physics("ec").feature("potZZ").set("V0","Vapp");
        model.component("comp1").physics("ec").create("gndZZ","Ground",2);
        model.component("comp1").physics("ec").feature("gndZZ").selection().named("selOutZZ");
        try{model.component("comp1").physics("ht").feature().remove("tempInZZ");}catch(Exception e){}
        try{model.component("comp1").physics("ht").feature().remove("tempOutZZ");}catch(Exception e){}
        model.component("comp1").physics("ht").create("tempInZZ","TemperatureBoundary",2);
        model.component("comp1").physics("ht").feature("tempInZZ").selection().named("selInZZ");
        model.component("comp1").physics("ht").feature("tempInZZ").set("T0","Telectrode");
        model.component("comp1").physics("ht").create("tempOutZZ","TemperatureBoundary",2);
        model.component("comp1").physics("ht").feature("tempOutZZ").selection().named("selOutZZ");
        model.component("comp1").physics("ht").feature("tempOutZZ").set("T0","Telectrode");

        // Solid Mechanics：两端面固定 + 热膨胀载荷，用于直接求解 solid.mises
        try{model.component("comp1").physics("solid").feature().remove("fixInZZ");}catch(Exception e){}
        try{model.component("comp1").physics("solid").feature().remove("fixOutZZ");}catch(Exception e){}
        model.component("comp1").physics("solid").create("fixInZZ","Fixed",2);
        model.component("comp1").physics("solid").feature("fixInZZ").selection().named("selInZZ");
        model.component("comp1").physics("solid").create("fixOutZZ","Fixed",2);
        model.component("comp1").physics("solid").feature("fixOutZZ").selection().named("selOutZZ");
        try {
            model.component("comp1").physics("solid").feature("lemm1").feature("te1");
        } catch(Exception e) {
            try {
                model.component("comp1").physics("solid").feature("lemm1").create("te1","ThermalExpansion",3);
            } catch(Exception e2) {}
        }
        try{model.component("comp1").physics("solid").feature("lemm1").feature("te1").set("T","T");}catch(Exception e){}
        try{model.component("comp1").physics("solid").feature("lemm1").feature("te1").set("Temp","T");}catch(Exception e){}
        try{model.component("comp1").physics("solid").feature("lemm1").feature("te1").set("minput_temperature","T");}catch(Exception e){}
        try{model.component("comp1").physics("solid").feature("lemm1").feature("te1").set("minput_temperature_src","root.comp1.T");}catch(Exception e){}
        try{model.component("comp1").physics("solid").feature("lemm1").feature("te1").set("Tref","Tamb");}catch(Exception e){}
        try{model.component("comp1").physics("solid").feature("lemm1").feature("te1").set("T0","Tamb");}catch(Exception e){}
        try{model.component("comp1").physics("solid").feature("lemm1").feature("te1").set("Tempref","Tamb");}catch(Exception e){}
        // 兼容部分 COMSOL 版本中 ThermalExpansion 作为 solid 顶层特征的情况。
        try {
            model.component("comp1").physics("solid").feature("teTopZZ");
        } catch(Exception e) {
            try {
                model.component("comp1").physics("solid").create("teTopZZ","ThermalExpansion",3);
            } catch(Exception e2) {}
        }
        try{model.component("comp1").physics("solid").feature("teTopZZ").set("T","T");}catch(Exception e){}
        try{model.component("comp1").physics("solid").feature("teTopZZ").set("Temp","T");}catch(Exception e){}
        try{model.component("comp1").physics("solid").feature("teTopZZ").set("minput_temperature","T");}catch(Exception e){}
        try{model.component("comp1").physics("solid").feature("teTopZZ").set("minput_temperature_src","root.comp1.T");}catch(Exception e){}
        try{model.component("comp1").physics("solid").feature("teTopZZ").set("Tref","Tamb");}catch(Exception e){}
        try{model.component("comp1").physics("solid").feature("teTopZZ").set("T0","Tamb");}catch(Exception e){}
        try{model.component("comp1").physics("solid").feature("teTopZZ").set("Tempref","Tamb");}catch(Exception e){}

        // S2S（MultipleSpectralBands，ε₀₃=0.35，ε_rest=0.15）
        model.component("comp1").physics().create("rad","SurfaceToSurfaceRadiation","geom1");
        model.component("comp1").physics("rad").prop("RadiationSettings").set("wavelengthDependenceOfSurfaceProperties","MultipleSpectralBands");
        model.component("comp1").physics("rad").prop("RadiationSettings").set("lambda_r","3");
        model.component("comp1").physics("rad").create("dsZZ","DiffuseSurface",2);
        model.component("comp1").physics("rad").feature("dsZZ").selection().all();
        model.component("comp1").physics("rad").feature("dsZZ").set("defineSurfaceEmissivityOnEachSide","0");
        model.component("comp1").physics("rad").feature("dsZZ").set("epsilon_radMulti_mat","userdef");
        model.component("comp1").physics("rad").feature("dsZZ").set("epsilon_radMulti","if(z<1e-9[m],0,if(z>L0-1e-9[m],0,if(comp1.rad.lambda<lam03,eps03,epsRest)))");
        model.component("comp1").physics("rad").feature("dsZZ").set("spectralBandNameAmbientEmissivityMulti",new String[][]{{"[0, 3["},{"[3, +inf["}});
        model.component("comp1").physics("rad").feature("dsZZ").set("Tamb","Tamb");
        model.component("comp1").physics("rad").feature("dsZZ").set("Tambu","Tamb");
        model.component("comp1").physics("rad").feature("dsZZ").set("Tambd","Tamb");
        model.component("comp1").physics("rad").feature("dsZZ").set("ambientEmissivity","userdef");
        model.component("comp1").physics("rad").feature("dsZZ").set("epsilon_amb","1");
        model.component("comp1").physics("rad").feature("dsZZ").set("epsilon_ambu","1");
        model.component("comp1").physics("rad").feature("dsZZ").set("epsilon_ambd","1");
        model.multiphysics().create("htradZZ","HeatTransferWithSurfaceToSurfaceRadiation","geom1",2);
        model.multiphysics("htradZZ").selection().all();

        // 网格
        model.component("comp1").mesh("mesh1").feature("size").set("hauto",5);
        try{model.component("comp1").mesh("mesh1").feature("ftet1");}
        catch(Exception e){model.component("comp1").mesh("mesh1").create("ftet1","FreeTet");}
        model.component("comp1").mesh("mesh1").run();
    }

    // ================================================================
    //  Phase 0: 1V sanity + 创建数值算子
    //  ★ 初始建模时清旧 solver（含 remove），之后只用 clearSolution ★
    // ================================================================
    try {
        for (String st : model.sol().tags()) {
            try{model.sol(st).clearSolution();}catch(Exception e2){}
            try{model.sol().remove(st);}catch(Exception e2){}
        }
    } catch(Exception e){}

    model.param().set("Vapp","1[V]");
    model.study("std1").run();

    // 数值算子（创建一次，始终引用同一 dataset）
    model.result().numerical().create("maxTZZ","MaxVolume");
    model.result().numerical("maxTZZ").selection().all();
    model.result().numerical("maxTZZ").set("expr",new String[]{"T"});
    model.result().numerical().create("minTZZ","MinVolume");
    model.result().numerical("minTZZ").selection().all();
    model.result().numerical("minTZZ").set("expr",new String[]{"T"});
    model.result().numerical().create("volZZ","IntVolume");
    model.result().numerical("volZZ").selection().all();
    model.result().numerical("volZZ").set("expr",new String[]{"1"});
    model.result().numerical().create("TintVolZZ","IntVolume");
    model.result().numerical("TintVolZZ").selection().all();
    model.result().numerical("TintVolZZ").set("expr",new String[]{"T"});
    model.result().numerical().create("IinZZ","IntSurface");
    model.result().numerical("IinZZ").selection().named("selInZZ");
    model.result().numerical("IinZZ").set("expr",new String[]{"ec.Jx*nx+ec.Jy*ny+ec.Jz*nz"});
    model.result().numerical().create("P03emitZZ","IntSurface");
    model.result().numerical("P03emitZZ").selection().all();
    model.result().numerical("P03emitZZ").set("expr",new String[]{"rad.epsilonu_band1*rad.ebu1"});
    model.result().numerical().create("PradEmitZZ","IntSurface");
    model.result().numerical("PradEmitZZ").selection().all();
    model.result().numerical("PradEmitZZ").set("expr",new String[]{"rad.epsilonu_band1*rad.ebu1+rad.epsilonu_band2*rad.ebu2"});
    model.result().numerical().create("P03escapeZZ","IntSurface");
    model.result().numerical("P03escapeZZ").selection().all();
    model.result().numerical("P03escapeZZ").set("expr",new String[]{"rad.J_band1*rad.Famb1"});
    model.result().numerical().create("PradEscapeZZ","IntSurface");
    model.result().numerical("PradEscapeZZ").selection().all();
    model.result().numerical("PradEscapeZZ").set("expr",new String[]{"rad.J_band1*rad.Famb1+rad.J_band2*rad.Famb2"});

    for(int bi=0;bi<blkCount;bi++){
        String tTag="TintBlk_"+(bi+1);
        String aTag="AblkLat_"+(bi+1);
        String sTag="selBlkLat_"+(bi+1);
        try{model.result().numerical().remove(tTag);}catch(Exception e){}
        model.result().numerical().create(tTag,"IntSurface");
        model.result().numerical(tTag).selection().named(sTag);
        model.result().numerical(tTag).set("expr",new String[]{"T"});
        try{model.result().numerical().remove(aTag);}catch(Exception e){}
        model.result().numerical().create(aTag,"IntSurface");
        model.result().numerical(aTag).selection().named(sTag);
        model.result().numerical(aTag).set("expr",new String[]{"1"});
    }

    double sanT=model.result().numerical("maxTZZ").getReal()[0][0];
    double sanI=Math.abs(model.result().numerical("IinZZ").getReal()[0][0]);
    System.out.println("SANITY: Tmax="+String.format("%.1f",sanT)+"K  I="+String.format("%.4f",sanI)+"A");
    if (sanT < 200) throw new RuntimeException("S2S coupling failure: Tmax="+sanT+"K");

    // ================================================================
    //  solveAt(V): 设电压 → clearSolution → study.run() → 提取 → 填 solRes[11]
    //  ★ 只 clearSolution，不 remove，dataset 引用保持有效 ★
    // ================================================================
    double[] solRes = new double[11];

    // ================================================================
    //  Phase 1: 电压搜索
    // ================================================================
    System.out.println("Phase 1: voltage search...");

    // 尝试 100V
    {
        model.param().set("Vapp",voltageUpper+"[V]");
        try{for(String st:model.sol().tags()) try{model.sol(st).clearSolution();}catch(Exception e2){}}catch(Exception e){}
        model.study("std1").run();
        double Tmx=model.result().numerical("maxTZZ").getReal()[0][0];
        double Tmn; try{Tmn=model.result().numerical("minTZZ").getReal()[0][0];}catch(Exception e){Tmn=Tmx*0.95;}
        double Vol=model.result().numerical("volZZ").getReal()[0][0];
        double Icur=Math.abs(model.result().numerical("IinZZ").getReal()[0][0]);
        double P03=model.result().numerical("P03emitZZ").getReal()[0][0];
        double Prad=model.result().numerical("PradEmitZZ").getReal()[0][0];
        double P03Sph=model.result().numerical("P03escapeZZ").getReal()[0][0];
        double PradSph=model.result().numerical("PradEscapeZZ").getReal()[0][0];
        double vErr=Math.abs(Vol-V0)/V0;
        solRes[0]=Tmx; solRes[1]=Tmn; solRes[2]=Icur; solRes[3]=P03; solRes[4]=Prad;
        solRes[5]=P03Sph; solRes[6]=PradSph; solRes[7]=vErr;
        solRes[8]=(Tmx<tempLimitK)?1:0; solRes[9]=(vErr<=volTol)?1:0; solRes[10]=(Icur>currentTol)?1:0;
    }
    System.out.println("  V="+voltageUpper+"V -> Tmax="+String.format("%.1f",solRes[0])+"K ok="+(solRes[8]>0&&solRes[9]>0&&solRes[10]>0));

    double Vwork; double[] r0Res=new double[11];
    if (solRes[8]>0 && solRes[9]>0 && solRes[10]>0) {
        Vwork=voltageUpper; System.arraycopy(solRes,0,r0Res,0,11);
    } else {
        double highV=voltageUpper; double[] highRes=new double[11]; System.arraycopy(solRes,0,highRes,0,11);
        double lowV=-1; double[] lowRes=null;

        if (solRes[10]>0&&solRes[9]>0) {
            double gV=voltageUpper*Math.sqrt(tempLimitK/Math.max(solRes[0],1e-300));
            gV=Math.max(voltageFloor,Math.min(0.98*voltageUpper,gV));
            if (gV<highV-1e-12) {
                model.param().set("Vapp",gV+"[V]");
                try{for(String st:model.sol().tags()) try{model.sol(st).clearSolution();}catch(Exception e2){}}catch(Exception e){}
                model.study("std1").run();
                double Tmx=model.result().numerical("maxTZZ").getReal()[0][0];
                double Tmn; try{Tmn=model.result().numerical("minTZZ").getReal()[0][0];}catch(Exception e){Tmn=Tmx*0.95;}
                double Vol=model.result().numerical("volZZ").getReal()[0][0];
                double Icur=Math.abs(model.result().numerical("IinZZ").getReal()[0][0]);
                double P03=model.result().numerical("P03emitZZ").getReal()[0][0];
                double Prad=model.result().numerical("PradEmitZZ").getReal()[0][0];
                double P03Sph=model.result().numerical("P03escapeZZ").getReal()[0][0];
                double PradSph=model.result().numerical("PradEscapeZZ").getReal()[0][0];
                double vErr=Math.abs(Vol-V0)/V0;
                double[] gRes={Tmx,Tmn,Icur,P03,Prad,P03Sph,PradSph,vErr,(Tmx<tempLimitK)?1:0,(vErr<=volTol)?1:0,(Icur>currentTol)?1:0};
                if(gRes[8]>0&&gRes[9]>0&&gRes[10]>0){lowV=gV;lowRes=gRes;}
                else{highV=gV;System.arraycopy(gRes,0,highRes,0,11);}
            }
        }
        while (lowRes==null && highV>voltageFloor+1e-12) {
            double nV=Math.max(voltageFloor,0.5*highV);
            if (Math.abs(nV-highV)<=1e-12) break;
            model.param().set("Vapp",nV+"[V]");
            try{for(String st:model.sol().tags()) try{model.sol(st).clearSolution();}catch(Exception e2){}}catch(Exception e){}
            model.study("std1").run();
            double Tmx=model.result().numerical("maxTZZ").getReal()[0][0];
            double Tmn; try{Tmn=model.result().numerical("minTZZ").getReal()[0][0];}catch(Exception e){Tmn=Tmx*0.95;}
            double Vol=model.result().numerical("volZZ").getReal()[0][0];
            double Icur=Math.abs(model.result().numerical("IinZZ").getReal()[0][0]);
            double P03=model.result().numerical("P03emitZZ").getReal()[0][0];
            double Prad=model.result().numerical("PradEmitZZ").getReal()[0][0];
            double P03Sph=model.result().numerical("P03escapeZZ").getReal()[0][0];
            double PradSph=model.result().numerical("PradEscapeZZ").getReal()[0][0];
            double vErr=Math.abs(Vol-V0)/V0;
            double[] nRes={Tmx,Tmn,Icur,P03,Prad,P03Sph,PradSph,vErr,(Tmx<tempLimitK)?1:0,(vErr<=volTol)?1:0,(Icur>currentTol)?1:0};
            if(nRes[8]>0&&nRes[9]>0&&nRes[10]>0){lowV=nV;lowRes=nRes;}
            else{highV=nV;System.arraycopy(nRes,0,highRes,0,11);}
        }
        if (lowRes==null) throw new RuntimeException("FAIL: voltage search failed");
        for (int vi=0; vi<maxVoltageIters; vi++) {
            if (highV-lowV<=voltageTol) break;
            double mV=0.5*(lowV+highV);
            model.param().set("Vapp",mV+"[V]");
            try{for(String st:model.sol().tags()) try{model.sol(st).clearSolution();}catch(Exception e2){}}catch(Exception e){}
            model.study("std1").run();
            double Tmx=model.result().numerical("maxTZZ").getReal()[0][0];
            double Tmn; try{Tmn=model.result().numerical("minTZZ").getReal()[0][0];}catch(Exception e){Tmn=Tmx*0.95;}
            double Vol=model.result().numerical("volZZ").getReal()[0][0];
            double Icur=Math.abs(model.result().numerical("IinZZ").getReal()[0][0]);
            double P03=model.result().numerical("P03emitZZ").getReal()[0][0];
            double Prad=model.result().numerical("PradEmitZZ").getReal()[0][0];
            double P03Sph=model.result().numerical("P03escapeZZ").getReal()[0][0];
            double PradSph=model.result().numerical("PradEscapeZZ").getReal()[0][0];
            double vErr=Math.abs(Vol-V0)/V0;
            double[] mRes={Tmx,Tmn,Icur,P03,Prad,P03Sph,PradSph,vErr,(Tmx<tempLimitK)?1:0,(vErr<=volTol)?1:0,(Icur>currentTol)?1:0};
            if(mRes[8]>0&&mRes[9]>0&&mRes[10]>0){lowV=mV;lowRes=mRes;}
            else highV=mV;
        }
        Vwork=lowV; System.arraycopy(lowRes,0,r0Res,0,11);
    }
    double r0Tmean=Double.NaN, r0U=Double.NaN;
    {
        model.param().set("Vapp",Vwork+"[V]");
        try{for(String st:model.sol().tags()) try{model.sol(st).clearSolution();}catch(Exception e2){}}catch(Exception e){}
        model.study("std1").run();
        double Tmx=model.result().numerical("maxTZZ").getReal()[0][0];
        double Tmn; try{Tmn=model.result().numerical("minTZZ").getReal()[0][0];}catch(Exception e){Tmn=Tmx*0.95;}
        double Vol=model.result().numerical("volZZ").getReal()[0][0];
        double TintVol=model.result().numerical("TintVolZZ").getReal()[0][0];
        double Icur=Math.abs(model.result().numerical("IinZZ").getReal()[0][0]);
        double P03=model.result().numerical("P03emitZZ").getReal()[0][0];
        double Prad=model.result().numerical("PradEmitZZ").getReal()[0][0];
        double vErr=Math.abs(Vol-V0)/V0;
        r0Tmean=(Vol>1e-20)?TintVol/Vol:Double.NaN;
        r0U=(r0Tmean>1e-20)?(Tmx-Tmn)/r0Tmean*100.0:Double.NaN;
        r0Res[0]=Tmx; r0Res[1]=Tmn; r0Res[2]=Icur; r0Res[3]=P03; r0Res[4]=Prad;
        r0Res[5]=P03; r0Res[6]=Prad; r0Res[7]=vErr;
        r0Res[8]=(Tmx<tempLimitK)?1:0; r0Res[9]=(vErr<=volTol)?1:0; r0Res[10]=(Icur>currentTol)?1:0;
    }
    System.out.println("PHASE1: Vwork="+String.format("%.4f",Vwork)+"V  Tmax="+String.format("%.1f",r0Res[0])+"K  P03sph="+String.format("%.1f",r0Res[5])+"W");

    // ---- 过程图像：0%、约 6.7%、约 13.3%、20% 损失四个节点 ----
    StageDatasetOps stageOps = new StageDatasetOps();
    double[] geomSideHist = new double[maxErosionSteps + 1];
    double[] shrinkHist = new double[maxErosionSteps + 1];
    double[] renvHist = new double[maxErosionSteps + 1];
    double[] timeHist = new double[maxErosionSteps + 1];
    double[] lossHist = new double[maxErosionSteps + 1];
    double[] tmaxHist = new double[maxErosionSteps + 1];
    boolean[] histOk = new boolean[maxErosionSteps + 1];
    geomSideHist[0] = side0;
    shrinkHist[0] = 0.0;
    renvHist[0] = Renv0;
    timeHist[0] = 0.0;
    lossHist[0] = 0.0;
    tmaxHist[0] = r0Res[0];
    histOk[0] = true;

    // ================================================================
    //  Phase 2: 侵蚀循环
    // ================================================================
    System.out.println("Phase 2: erosion loop...");
    double timeS=0.0, p03Int=0.0, pradInt=0.0, p03sInt=0.0, pradsInt=0.0;
    int macro=0; boolean failed=false;
    double resolveThr=0.02*side0, sideMin=side0*(1.0-failureFraction);
    double[] blockSides=new double[blkCount]; for(int bi=0;bi<blkCount;bi++) blockSides[bi]=side0;

    double[] blockTavg=new double[blkCount];
    for(int bi=0;bi<blkCount;bi++){
        boolean readOk=false;
        try{
            double Tint=model.result().numerical("TintBlk_"+(bi+1)).getReal()[0][0];
            double Ablk=model.result().numerical("AblkLat_"+(bi+1)).getReal()[0][0];
            if(Ablk>1e-20){blockTavg[bi]=Tint/Ablk; readOk=true;}
        }catch(Exception e){}
        if(!readOk){
            double zc=(blkIsHoriz[bi])?(blkZ0[bi]+0.5*side0):(blkZ0[bi]+0.5*blkSZ[bi]);
            blockTavg[bi]=r0Res[1]+(r0Res[0]-r0Res[1])*4.0*(zc/L0Value)*(1.0-zc/L0Value);
        }
    }
    double prevP03=r0Res[3], prevPrad=r0Res[4], prevP03s=r0Res[5], prevPrads=r0Res[6];
    double maxErosionTmax=r0Res[0];
    int overtempStep=-1;
    double overtempTimeH=Double.NaN;
    double overtempTmax=Double.NaN;
    String status="OK";

    while (macro<maxErosionSteps && !failed) {
        macro++;

        double[] dsdt=new double[blkCount]; double maxDsdt=0.0;
        for(int bi=0;bi<blkCount;bi++){
            dsdt[bi]=2.0*AevVal*Math.exp(-BevVal/blockTavg[bi])/rhoMassWValue;
            if(dsdt[bi]>maxDsdt) maxDsdt=dsdt[bi];
        }
        if(maxDsdt<1e-15){System.out.println("  Evaporation negligible.");break;}

        double dt=Double.MAX_VALUE;
        for(int bi=0;bi<blkCount;bi++) if(dsdt[bi]>1e-20){
            dt=Math.min(dt,resolveThr/dsdt[bi]);
            double tF=(blockSides[bi]-sideMin)/dsdt[bi]; if(tF>0) dt=Math.min(dt,tF);
        }
        dt=Math.max(1.0,Math.min(36000.0,dt));

        double maxLoss=0.0;
        for(int bi=0;bi<blkCount;bi++){
            blockSides[bi]=Math.max(1e-6,blockSides[bi]-dsdt[bi]*dt);
            double loss=(side0-blockSides[bi])/side0; if(loss>maxLoss) maxLoss=loss;
        }
        timeS+=dt;
        if(maxLoss>=failureFraction) failed=true;

        // 均匀侵蚀近似
        double geomSide=Double.POSITIVE_INFINITY;
        for(double s:blockSides) if(s<geomSide) geomSide=s;
        if(Double.isInfinite(geomSide)) geomSide=side0;
        double shrink=(side0-geomSide)*0.5;

        // 更新外接球参数
        double md3=0.0, zc3=0.5*L0Value;
        for(int bi=0;bi<blkCount;bi++){
            double ex0,ex1,ey0,ey1,ez0,ez1;
            if(blkIsHoriz[bi]){ex0=blkX0[bi];ex1=blkX0[bi]+blkSX[bi];ey0=blkY0[bi]+shrink;ey1=ey0+geomSide;ez0=blkZ0[bi]+shrink;ez1=ez0+geomSide;}
            else{ex0=blkX0[bi]+shrink;ex1=ex0+geomSide;ey0=blkY0[bi]+shrink;ey1=ey0+geomSide;ez0=blkZ0[bi];ez1=blkZ0[bi]+blkSZ[bi];}
            for(double xx:new double[]{ex0,ex1}) for(double yy:new double[]{ey0,ey1}) for(double zz:new double[]{ez0,ez1}){
                double d3=Math.sqrt(xx*xx+yy*yy+(zz-zc3)*(zz-zc3)); if(d3>md3) md3=d3;
            }
        }
        for(double rx:new double[]{-r0Value,r0Value}) for(double ry:new double[]{-r0Value,r0Value})
            for(double zz:new double[]{0,terminalStubLen,L0Value-terminalStubLen,L0Value}){
                double d3=Math.sqrt(rx*rx+ry*ry+(zz-zc3)*(zz-zc3)); if(d3>md3) md3=d3;
            }
        double newRenv=outerSphereMargin*md3;
        model.param().set("RenvZZ",Double.toString(newRenv)+"[m]");
        model.param().set("AenvZZ",Double.toString(4*Math.PI*newRenv*newRenv)+"[m^2]");

        // ★ 只重建几何 + 网格，不重建物理场（保留 solver/dataset 引用）★
        boolean rebuildOk=true;
        try {
            try{model.component("comp1").geom("geom1").feature().remove("uniZZ");}catch(Exception e){}
            try{model.component("comp1").geom("geom1").feature().remove("term_in");}catch(Exception e){}
            try{model.component("comp1").geom("geom1").feature().remove("term_out");}catch(Exception e){}
            for(int bi=0;bi<MAX_BLOCK_SLOTS;bi++) try{model.component("comp1").geom("geom1").feature().remove("blk_"+(bi+1));}catch(Exception e){}

            java.util.ArrayList<String> tl=new java.util.ArrayList<String>();
            for(int bi=0;bi<blkCount;bi++){
                String tag="blk_"+(bi+1); tl.add(tag);
                double nx0,ny0,nz0,nsx,nsy,nsz;
                if(blkIsHoriz[bi]){nx0=blkX0[bi];ny0=blkY0[bi]+shrink;nz0=blkZ0[bi]+shrink;nsx=blkSX[bi];nsy=geomSide;nsz=geomSide;}
                else{nx0=blkX0[bi]+shrink;ny0=blkY0[bi]+shrink;nz0=blkZ0[bi];nsx=geomSide;nsy=geomSide;nsz=blkSZ[bi];}
                model.component("comp1").geom("geom1").create(tag,"Block");
                model.component("comp1").geom("geom1").feature(tag).set("size",new String[]{Double.toString(nsx)+"[m]",Double.toString(nsy)+"[m]",Double.toString(nsz)+"[m]"});
                model.component("comp1").geom("geom1").feature(tag).set("pos",new String[]{Double.toString(nx0)+"[m]",Double.toString(ny0)+"[m]",Double.toString(nz0)+"[m]"});
            }
            model.component("comp1").geom("geom1").create("term_in","Cylinder");
            model.component("comp1").geom("geom1").feature("term_in").set("r",Double.toString(r0Value)+"[m]");
            model.component("comp1").geom("geom1").feature("term_in").set("h",Double.toString(terminalStubLen)+"[m]");
            model.component("comp1").geom("geom1").feature("term_in").set("pos",new String[]{"0[m]","0[m]","0[m]"});
            model.component("comp1").geom("geom1").create("term_out","Cylinder");
            model.component("comp1").geom("geom1").feature("term_out").set("r",Double.toString(r0Value)+"[m]");
            model.component("comp1").geom("geom1").feature("term_out").set("h",Double.toString(terminalStubLen)+"[m]");
            model.component("comp1").geom("geom1").feature("term_out").set("pos",new String[]{"0[m]","0[m]",Double.toString(L0Value-terminalStubLen)+"[m]"});
            String[] ut=new String[blkCount+2];
            for(int bi=0;bi<blkCount;bi++) ut[bi]=tl.get(bi);
            ut[blkCount]="term_in"; ut[blkCount+1]="term_out";
            model.component("comp1").geom("geom1").create("uniZZ","Union");
            model.component("comp1").geom("geom1").feature("uniZZ").selection("input").set(ut);
            model.component("comp1").geom("geom1").feature("uniZZ").set("intbnd",false);
            model.component("comp1").geom("geom1").run();

            for(int bi=0;bi<blkCount;bi++){
                String sTag="selBlkLat_"+(bi+1);
                double nx0,ny0,nz0,nsx,nsy,nsz;
                if(blkIsHoriz[bi]){nx0=blkX0[bi];ny0=blkY0[bi]+shrink;nz0=blkZ0[bi]+shrink;nsx=blkSX[bi];nsy=geomSide;nsz=geomSide;}
                else{nx0=blkX0[bi]+shrink;ny0=blkY0[bi]+shrink;nz0=blkZ0[bi];nsx=geomSide;nsy=geomSide;nsz=blkSZ[bi];}
                double pad=Math.max(1e-5,0.01*geomSide*1e3);
                try{model.component("comp1").selection(sTag);}catch(Exception e){
                    model.component("comp1").selection().create(sTag,"Box");
                    model.component("comp1").selection(sTag).geom("geom1",2);
                    model.component("comp1").selection(sTag).set("condition","intersects");
                }
                if(blkIsHoriz[bi]){
                    double x0=(nx0+0.10*nsx)*1e3, x1=(nx0+0.90*nsx)*1e3;
                    if(x1<=x0){x0=nx0*1e3; x1=(nx0+nsx)*1e3;}
                    model.component("comp1").selection(sTag).set("xmin",x0);
                    model.component("comp1").selection(sTag).set("xmax",x1);
                    model.component("comp1").selection(sTag).set("ymin",ny0*1e3-pad);
                    model.component("comp1").selection(sTag).set("ymax",(ny0+nsy)*1e3+pad);
                    model.component("comp1").selection(sTag).set("zmin",nz0*1e3-pad);
                    model.component("comp1").selection(sTag).set("zmax",(nz0+nsz)*1e3+pad);
                } else {
                    double z0=(nz0+0.10*nsz)*1e3, z1=(nz0+0.90*nsz)*1e3;
                    if(z1<=z0){z0=nz0*1e3; z1=(nz0+nsz)*1e3;}
                    model.component("comp1").selection(sTag).set("xmin",nx0*1e3-pad);
                    model.component("comp1").selection(sTag).set("xmax",(nx0+nsx)*1e3+pad);
                    model.component("comp1").selection(sTag).set("ymin",ny0*1e3-pad);
                    model.component("comp1").selection(sTag).set("ymax",(ny0+nsy)*1e3+pad);
                    model.component("comp1").selection(sTag).set("zmin",z0);
                    model.component("comp1").selection(sTag).set("zmax",z1);
                }
            }

            model.component("comp1").mesh("mesh1").feature("size").set("hauto",5);
            try{model.component("comp1").mesh("mesh1").feature("ftet1");}
            catch(Exception e){model.component("comp1").mesh("mesh1").create("ftet1","FreeTet");}
            model.component("comp1").mesh("mesh1").run();

        } catch(Exception rebuildEx) {
            System.out.println("  WARN rebuild step "+macro+": "+rebuildEx.getMessage());
            rebuildOk=false; failed=true;
            status="FAIL_EROSION_REBUILD";
        }

        double curP03=prevP03, curPrad=prevPrad, curP03s=prevP03s, curPrads=prevPrads;
        double curTmax=0.0, curTmin=0.0;
        if (rebuildOk) {
            try {
                // ★ 只 clearSolution，保持 dataset 引用有效 ★
                try{for(String st2:model.sol().tags()) try{model.sol(st2).clearSolution();}catch(Exception e3){}}catch(Exception e2){}
                model.param().set("Vapp",Vwork+"[V]");
                model.study("std1").run();
                curTmax=model.result().numerical("maxTZZ").getReal()[0][0];
                try{curTmin=model.result().numerical("minTZZ").getReal()[0][0];}catch(Exception e){curTmin=curTmax*0.95;}
                double Ic=Math.abs(model.result().numerical("IinZZ").getReal()[0][0]);
                curP03=model.result().numerical("P03emitZZ").getReal()[0][0];
                curPrad=model.result().numerical("PradEmitZZ").getReal()[0][0];
                curP03s=model.result().numerical("P03escapeZZ").getReal()[0][0];
                curPrads=model.result().numerical("PradEscapeZZ").getReal()[0][0];
            } catch(Exception solEx) {
                System.out.println("  WARN solve step "+macro+": "+solEx.getMessage());
                failed=true;
                status="FAIL_EROSION_SOLVE";
            }
        }

        p03Int  +=0.5*(prevP03  +curP03 )*dt; pradInt +=0.5*(prevPrad +curPrad )*dt;
        p03sInt +=0.5*(prevP03s +curP03s)*dt; pradsInt+=0.5*(prevPrads+curPrads)*dt;
        prevP03=curP03; prevPrad=curPrad; prevP03s=curP03s; prevPrads=curPrads;

        if (rebuildOk&&curTmax>0) for(int bi=0;bi<blkCount;bi++){
            boolean readOk=false;
            try{
                double Tint=model.result().numerical("TintBlk_"+(bi+1)).getReal()[0][0];
                double Ablk=model.result().numerical("AblkLat_"+(bi+1)).getReal()[0][0];
                if(Ablk>1e-20){blockTavg[bi]=Tint/Ablk; readOk=true;}
            }catch(Exception e){}
            if(!readOk){
                double zc=(blkIsHoriz[bi])?(blkZ0[bi]+0.5*side0):(blkZ0[bi]+0.5*blkSZ[bi]);
                double eta=zc/L0Value;
                blockTavg[bi]=curTmin+(curTmax-curTmin)*4.0*eta*(1.0-eta);
            }
        }

        if (rebuildOk&&curTmax>maxErosionTmax) maxErosionTmax=curTmax;

        geomSideHist[macro] = geomSide;
        shrinkHist[macro] = shrink;
        renvHist[macro] = newRenv;
        timeHist[macro] = timeS;
        lossHist[macro] = maxLoss;
        tmaxHist[macro] = curTmax;
        histOk[macro] = rebuildOk;

        if (rebuildOk&&curTmax>=tempLimitK) {
            status="FAIL_OVERTEMP_DURING_EROSION";
            overtempStep=macro;
            overtempTimeH=timeS/3600.0;
            overtempTmax=curTmax;
            break;
        }

        if(macro%5==0||failed)
            System.out.println("  STEP="+macro+" t="+String.format("%.2f",timeS/3600.0)+"h loss="+String.format("%.4f",maxLoss));
    }

    // ================================================================
    //  Phase 3: 输出
    // ================================================================
    double lifetimeH=timeS/3600.0;
    double avgP03gross=(timeS>0)?p03Int/timeS:Double.NaN;
    double avgP03s=(timeS>0)?p03sInt/timeS:Double.NaN;
    double avgPrads=(timeS>0)?pradsInt/timeS:Double.NaN;
    double svLossRaw=(p03Int>0)?(1.0-p03sInt/p03Int)*100.0:Double.NaN;
    double svLoss=Double.isNaN(svLossRaw)?Double.NaN:Math.max(0.0,svLossRaw);

    System.out.println("============================================================");
    System.out.println("  ZIGZAG BASELINE RESULT  (MultipleSpectralBands S2S)");
    System.out.println("============================================================");
    System.out.println("  Vwork              = "+String.format("%.4f",Vwork)+" V");
    System.out.println("  initialTmax_K      = "+String.format("%.1f",r0Res[0])+" K");
    System.out.println("  Tmin_K             = "+String.format("%.1f",r0Res[1])+" K");
    System.out.println("  Tmean_K            = "+String.format("%.1f",r0Tmean)+" K");
    System.out.println("  U_pct              = "+String.format("%.4f",r0U)+"%");
    System.out.println("  maxErosionTmax_K   = "+String.format("%.1f",maxErosionTmax)+" K");
    System.out.println("  lifetimeH          = "+String.format("%.4f",lifetimeH)+" h");
    System.out.println("  initialP03sphere_W = "+String.format("%.2f",r0Res[5])+" W");
    System.out.println("  initialPrad_sphere = "+String.format("%.2f",r0Res[6])+" W");
    System.out.println("  lifeAvgP03sph_W    = "+String.format("%.2f",avgP03s)+" W");
    System.out.println("  lifeAvgPradSph_W   = "+String.format("%.2f",avgPrads)+" W");
    System.out.println("  selfViewLoss_pct   = "+String.format("%.2f",svLoss)+"%");
    System.out.println("  failureReached     = "+failed);
    System.out.println("  erosionSteps       = "+macro);
    System.out.println("  status             = "+status);
    System.out.println("============================================================");

    System.out.println("RESULT_HEADER=Vwork_V,initialTmax_K,Tmin_K,Tmean_K,U_pct,maxErosionTmax_K,lifetimeH,initialP03sphere_W,initialPradSphere_W,lifeAvgP03sphere_W,lifeAvgPradSphere_W,lifeTotalP03sphere_J,selfViewLoss_pct,failureReached,erosionSteps,overtempStep,overtempTimeH,overtempTmax_K,status,metricVersion,initialP03gross_W,initialP03escape_W,initialP03selfAbsorbed_W,lifeAvgP03gross_W,lifeAvgP03escape_W,lifeTotalP03gross_J,lifeTotalP03escape_J,selfViewLossRaw_pct");
    System.out.println("RESULT="
        +String.format("%.4f",Vwork)+","
        +String.format("%.1f",r0Res[0])+","
        +String.format("%.1f",r0Res[1])+","
        +String.format("%.1f",r0Tmean)+","
        +String.format("%.4f",r0U)+","
        +String.format("%.1f",maxErosionTmax)+","
        +String.format("%.4f",lifetimeH)+","
        +String.format("%.2f",r0Res[5])+","
        +String.format("%.2f",r0Res[6])+","
        +String.format("%.2f",avgP03s)+","
        +String.format("%.2f",avgPrads)+","
        +String.format("%.2f",p03sInt)+","
        +String.format("%.2f",svLoss)+","
        +failed+","+macro+","
        +overtempStep+","
        +String.format("%.4f",overtempTimeH)+","
        +String.format("%.1f",overtempTmax)+","
        +status+",radiation_escape_v2,"
            +String.format("%.2f",r0Res[3])+","+String.format("%.2f",r0Res[5])+","
        +String.format("%.2f",Math.max(0.0,r0Res[3]-r0Res[5]))+","
        +String.format("%.2f",avgP03gross)+","+String.format("%.2f",avgP03s)+","
        +String.format("%.2f",p03Int)+","+String.format("%.2f",p03sInt)+","
        +String.format("%.6f",svLossRaw));

    // ================================================================
    // Phase 4: create four independent stage datasets in COMSOL Results.
    // Open zigzag_stage_01..04 and switch the Surface expression between
    // T and solid.mises as needed.
    // ================================================================
    int lastRecordedStep = macro;
    while (lastRecordedStep > 0 && !histOk[lastRecordedStep]) {
        lastRecordedStep--;
    }

    double finalLoss = lossHist[lastRecordedStep];
    double[] targetLosses = new double[]{0.0, finalLoss/3.0, 2.0*finalLoss/3.0, finalLoss};
    int[] keySteps = new int[4];
    for(int stageIdx=0;stageIdx<4;stageIdx++){
        int bestStep=0;
        double bestErr=Double.POSITIVE_INFINITY;
        for(int s=0;s<=lastRecordedStep;s++){
            if(!histOk[s]) continue;
            double err=Math.abs(lossHist[s]-targetLosses[stageIdx]);
            if(err<bestErr){bestErr=err; bestStep=s;}
        }
        keySteps[stageIdx]=(stageIdx==3)?lastRecordedStep:bestStep;
    }

    System.out.println("VIZ: Creating zigzag_stage_01..04 independent datasets.");
    for(int stageIdx=0;stageIdx<4;stageIdx++){
        int ks=keySteps[stageIdx];
        String idx=(stageIdx+1<10?"0":"")+(stageIdx+1);
        String stageName="zigzag_stage_"+idx;
        stageOps.rebuildStageGeometry(geomSideHist[ks],shrinkHist[ks],renvHist[ks]);
        model.param().set("Vapp",Vwork+"[V]");

        String stTag="stdZigzagStage"+idx;
        try{model.study().remove(stTag);}catch(Exception e){}
        model.study().create(stTag);
        model.study(stTag).label(stageName);
        model.study(stTag).create("stat","Stationary");
        model.study(stTag).run();

        String[] dsetTags=model.result().dataset().tags();
        String dsetTag=dsetTags[dsetTags.length-1];
        String pgTag="pgZigzagStage"+idx;
        String title=stageName
            +" | t="+String.format("%.3f",timeHist[ks]/3600.0)+" h"
            +" | loss="+String.format("%.2f",lossHist[ks]*100.0)+"%"
            +" | dataset="+dsetTag
            +" | edit expr: T or solid.mises";

        try{model.result().remove(pgTag);}catch(Exception e){}
        model.result().create(pgTag,"PlotGroup3D");
        model.result(pgTag).set("data",dsetTag);
        model.result(pgTag).label(stageName+" selectable field");
        model.result(pgTag).set("titletype","manual");
        model.result(pgTag).set("title",title);
        model.result(pgTag).create("surf1","Surface");
        model.result(pgTag).feature("surf1").set("expr","T");
        model.result(pgTag).feature("surf1").set("unit","K");
        model.result(pgTag).feature("surf1").set("colortable","ThermalLight");
        model.result(pgTag).run();

        System.out.println("VIZ_STAGE="+stageName
            +",studyTag="+stTag
            +",plotGroup="+pgTag
            +",dataset="+dsetTag
            +",step="+ks
            +",timeH="+String.format("%.6f",timeHist[ks]/3600.0)
            +",loss="+String.format("%.6f",lossHist[ks])
            +",exprOptions=T|solid.mises");
    }
    System.out.println("VIZ: Open zigzag_stage_01..04.  Change Surface expression to T or solid.mises.");
}
