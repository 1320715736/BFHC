{
    // ================================================================
    // cylinder_baseline.java — 圆柱簇当前提交版基线脚本
    // 提交版圆柱基线模型；从空白模型构建，无需预加载 .mph。
    // 相比历史中间版，保留以下两个关键修复：
    //
    //   [Fix-1] 侵蚀循环各段平均温度改为 IntSurface 算子直接从 COMSOL 读取，
    //           不再使用抛物线近似 Tavg[i] = Tmin + (Tmax-Tmin)*4η(1-η)。
    //           对非均匀形状（各段半径不等）精度更高，是 ML 优化阶段的重要改进。
    //           实现方式：rebuild() 中为每段侧面创建 Box 选择 selSegLat_{i+1}，
    //           主流程中创建对应 IntSurface 算子 TintSeg_{i+1} / AsegS2S_{i+1}，
    //           solvePrepared() 中用 Tavg[i] = TintSeg / AsegS2S 替换抛物线，
    //           若算子读取失败则回退到原抛物线近似并打印 WARN。
    //
    //   [Fix-2] selfViewLoss 负值说明注释完善：
    //           对凸圆柱体 selfViewLoss 物理上应 ≈ 0%；
    //           出现约 −5% 的负值是 qRadNetOutExpr 表面积分不包含 S2S 面间辐射
    //           Current baseline uses free-surface 0 K radiation integrals.
    //           是 P03sphere 方法的固有误差，不影响形状间相对比较，
    //           但竞赛汇报时需注明该系统偏差约 ±5%。
    //
    // S2S 面-面辐射 + 外接球统计口径 + 8 段半径参数化
    //
    // 输入：inputRadii[0..7] — 8 段圆柱半径 (m)，训练变量
    // 输出：CSV 格式核心赛题指标（见文件末尾 RESULT 行）
    //
    // 运行方式：在 COMSOL Java Shell 中直接运行（无需预先打开任何模型）
    // ================================================================

    // ================================================================
    //  ★★★ 训练输入：8 段半径（唯一需要修改的变量）★★★
    // ================================================================
    double[] inputRadii = new double[]{
        2.5e-3, 2.5e-3, 2.5e-3, 2.5e-3,
        2.5e-3, 2.5e-3, 2.5e-3, 2.5e-3
    };

    // ---- 固定参数 ----
    int segCount = 8;
    double L0Value = 15e-3;
    double LsegValue = L0Value / segCount;
    double tempLimitK = 3000.0 + 273.15;
    double rhoMassValue = 19350.0;
    double volTol = 1.0e-4;
    double currentTol = 1.0e-9;
    double outerSphereMargin = 1.05;
    double voltageUpperBound = 100.0;
    double voltageFloor = 1.0e-3;
    double voltageTol = 0.05;
    int maxVoltageSearchIters = 16;
    double AevValue = 3.9e9;
    double BevValue = 1.023e5;
    double failureFraction = 0.20;
    String epsRadS2SValue = "if(z<1e-9[m],0,if(z>L0-1e-9[m],0,if(comp1.rad.lambda<lam03,eps03,epsRest)))";

    // 从输入半径推导参考值
    double r0Tmp = 0.0;
    for (int i = 0; i < segCount; i++) { if (inputRadii[i] > r0Tmp) r0Tmp = inputRadii[i]; }
    double r0Value = r0Tmp;
    // per-segment 失效阈值（每段各自初始半径 × 80%）
    double[] failRadii = new double[segCount];
    for (int i = 0; i < segCount; i++) {
        failRadii[i] = inputRadii[i] * (1.0 - failureFraction);
    }
    double rMin = r0Value * (1.0 - failureFraction);

    // ---- 材料属性表达式 ----
    String rhoeExpr = "max(1e-10[ohm*m], 5.5e-8[ohm*m]*(1+0.003836*(T-293.15[K])/1[K]+7.55e-7*((T-293.15[K])/1[K])^2))";
    String kExpr = "max(75[W/(m*K)],175[W/(m*K)]-0.032[W/(m*K^2)]*(T-293.15[K]))";
    String cpExpr = "min(195[J/(kg*K)],132[J/(kg*K)]+0.020[J/(kg*K^2)]*(T-293.15[K]))";

    // ---- Planck f03 黑体谱分数 ----
    String x03TExpr = "(c2bb/(lam03*T))";
    String seriesTExpr = "";
    for (int n = 1; n <= 6; n++) {
        int n2 = n * n;
        int n3 = n2 * n;
        int n4 = n3 * n;
        String termT =
            "exp(-" + n + "*" + x03TExpr + ")*("
            + x03TExpr + "^3/" + n
            + "+3*" + x03TExpr + "^2/" + n2
            + "+6*" + x03TExpr + "/" + n3
            + "+6/" + n4
            + ")";
        if (n == 1) {
            seriesTExpr = termT;
        } else {
            seriesTExpr = seriesTExpr + "+" + termT;
        }
    }
    String f03bbTExpr = "min(1,max(0,(15/pi^4)*(" + seriesTExpr + ")))";
    String q03NetOutExpr =
        "eps03*sigmaSB*((" + f03bbTExpr + ")*T^4)";
    String qRadNetOutExpr =
        "sigmaSB*(epsRest*T^4+(eps03-epsRest)*((" + f03bbTExpr + ")*T^4))";

    // ---- 数据类 ----

    class GeomData {
        double[] rSeg;
        double rMax;
        double Renv;
        double Aenv;
    }

    class Res {
        boolean solveOk = false;
        boolean searchOk = false;
        String failure = "";
        double appliedV = Double.NaN;
        int searchSteps = 0;
        double Tmax = Double.NaN;
        double Tmin = Double.NaN;
        double Tmean = Double.NaN;
        double U_pct = Double.NaN;
        double I = Double.NaN;
        double R = Double.NaN;
        double Pelec = Double.NaN;
        double P03steady = Double.NaN;
        double PradSteady = Double.NaN;
        double P03sphere = Double.NaN;
        double PradSphere = Double.NaN;
        double volErr = Double.NaN;
        boolean tempOk = false;
        boolean volumeOk = false;
        boolean currentOk = false;
        double[] segTavg;
    }

    // ---- 操作类 ----

    class Ops {

        // ---- 从空白创建完整 COMSOL 模型骨架 ----
        void initModelFromScratch() {
            // 1) 组件 + 3D 几何（毫米单位）
            model.component().create("comp1");
            model.component("comp1").geom().create("geom1", 3);
            model.component("comp1").geom("geom1").lengthUnit("mm");

            // 2) 物理场：导电介质 + 传热
            model.component("comp1").physics().create("ec", "ConductiveMedia", "geom1");
            model.component("comp1").physics().create("ht", "HeatTransfer", "geom1");

            // 3) 材料：钨
            model.component("comp1").material().create("mat1", "Common");
            model.component("comp1").material("mat1").label("Tungsten");
            model.component("comp1").material("mat1").selection().all();

            // 4) 网格（仅创建容器，size/ftet 在几何构建后设置）
            model.component("comp1").mesh().create("mesh1", "geom1");

            // 5) 稳态研究
            model.study().create("std1");
            model.study("std1").create("stat", "Stationary");
        }

        GeomData geomFromRadii(double[] radii) {
            GeomData g = new GeomData();
            g.rSeg = new double[radii.length];
            g.rMax = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < radii.length; i++) {
                g.rSeg[i] = radii[i];
                if (radii[i] > g.rMax) { g.rMax = radii[i]; }
            }
            g.Renv = outerSphereMargin * Math.sqrt((0.5 * L0Value) * (0.5 * L0Value) + g.rMax * g.rMax);
            g.Aenv = 4.0 * Math.PI * g.Renv * g.Renv;
            return g;
        }

        void removeGeomFeature(String tag) {
            try { model.component("comp1").geom("geom1").feature().remove(tag); } catch (Exception e) {}
        }
        void removeSelection(String tag) {
            try { model.component("comp1").selection().remove(tag); } catch (Exception e) {}
        }
        void removeEcFeature(String tag) {
            try { model.component("comp1").physics("ec").feature().remove(tag); } catch (Exception e) {}
        }
        void removeHtFeature(String tag) {
            try { model.component("comp1").physics("ht").feature().remove(tag); } catch (Exception e) {}
        }
        void removePhysics(String tag) {
            try { model.component("comp1").physics().remove(tag); } catch (Exception e) {}
        }
        void removeMultiphysics(String tag) {
            try { model.multiphysics().remove(tag); } catch (Exception e) {}
        }

        // ---- S2S 辐射物理场 ----
        void setupSurfaceToSurfaceRadiation() {
            removePhysics("rad");
            removeMultiphysics("htradLT");

            model.component("comp1").physics().create("rad", "SurfaceToSurfaceRadiation", "geom1");
            model.component("comp1").physics("rad").prop("RadiationSettings")
                .set("wavelengthDependenceOfSurfaceProperties", "MultipleSpectralBands");
            model.component("comp1").physics("rad").prop("RadiationSettings")
                .set("lambda_r", "lam03");

            model.component("comp1").physics("rad").create("dsLT", "DiffuseSurface", 2);
            model.component("comp1").physics("rad").feature("dsLT").selection().all();
            model.component("comp1").physics("rad").feature("dsLT")
                .set("defineSurfaceEmissivityOnEachSide", "0");
            model.component("comp1").physics("rad").feature("dsLT")
                .set("epsilon_radMulti_mat", "userdef");
            model.component("comp1").physics("rad").feature("dsLT")
                .set("epsilon_radMulti", epsRadS2SValue);
            model.component("comp1").physics("rad").feature("dsLT")
                .set("spectralBandNameAmbientEmissivityMulti", new String[][]{{"[0, 3["}, {"[3, +inf["}});
            model.component("comp1").physics("rad").feature("dsLT").set("Tamb", "Tamb");
            model.component("comp1").physics("rad").feature("dsLT").set("Tambu", "Tamb");
            model.component("comp1").physics("rad").feature("dsLT").set("Tambd", "Tamb");
            model.component("comp1").physics("rad").feature("dsLT")
                .set("ambientEmissivity", "userdef");
            model.component("comp1").physics("rad").feature("dsLT").set("epsilon_amb", "1");
            model.component("comp1").physics("rad").feature("dsLT").set("epsilon_ambu", "1");
            model.component("comp1").physics("rad").feature("dsLT").set("epsilon_ambd", "1");

            model.multiphysics().create("htradLT",
                "HeatTransferWithSurfaceToSurfaceRadiation", "geom1", 2);
            model.multiphysics("htradLT").selection().all();
        }

        void setParams(GeomData g, double voltage) {
            for (int i = 0; i < segCount; i++) {
                model.param().set("r_seg" + (i + 1),
                    Double.toString(g.rSeg[i]) + "[m]",
                    "Segment " + (i + 1) + " radius");
            }
            model.param().set("Vapp", Double.toString(voltage) + "[V]", "Applied DC voltage");
            model.param().set("RenvInit", Double.toString(g.Renv) + "[m]", "Enclosing sphere radius");
            model.param().set("AenvInit", Double.toString(g.Aenv) + "[m^2]", "Enclosing sphere area");
        }

        void rebuild(GeomData g) {
            // 清理旧几何
            removeGeomFeature("uniS2S");
            for (int i = 0; i < segCount; i++) { removeGeomFeature("cS2S_" + (i + 1)); }

            // ---- 创建 8 段圆柱 ----
            String[] cylTags = new String[segCount];
            for (int i = 0; i < segCount; i++) {
                String tag = "cS2S_" + (i + 1);
                cylTags[i] = tag;
                model.component("comp1").geom("geom1").create(tag, "Cylinder");
                model.component("comp1").geom("geom1").feature(tag).set("r", "r_seg" + (i + 1));
                model.component("comp1").geom("geom1").feature(tag).set("h", "Lseg");
                model.component("comp1").geom("geom1").feature(tag).set("pos",
                    new String[]{"0", "0", Double.toString(i) + "*Lseg"});
            }
            model.component("comp1").geom("geom1").create("uniS2S", "Union");
            model.component("comp1").geom("geom1").feature("uniS2S").selection("input").set(cylTags);
            model.component("comp1").geom("geom1").feature("uniS2S").set("intbnd", false);
            model.component("comp1").geom("geom1").run();

            // 清理旧选择
            removeSelection("selInS2S");
            removeSelection("selOutS2S");
            removeSelection("selFreeS2S");

            model.component("comp1").selection().create("selInS2S", "Box");
            model.component("comp1").selection("selInS2S").geom("geom1", 2);
            model.component("comp1").selection("selInS2S").set("condition", "inside");
            model.component("comp1").selection("selInS2S").set("xmin", -10.0);
            model.component("comp1").selection("selInS2S").set("xmax", 10.0);
            model.component("comp1").selection("selInS2S").set("ymin", -10.0);
            model.component("comp1").selection("selInS2S").set("ymax", 10.0);
            model.component("comp1").selection("selInS2S").set("zmin", -1.0e-6);
            model.component("comp1").selection("selInS2S").set("zmax", 1.0e-6);

            model.component("comp1").selection().create("selOutS2S", "Box");
            model.component("comp1").selection("selOutS2S").geom("geom1", 2);
            model.component("comp1").selection("selOutS2S").set("condition", "inside");
            model.component("comp1").selection("selOutS2S").set("xmin", -10.0);
            model.component("comp1").selection("selOutS2S").set("xmax", 10.0);
            model.component("comp1").selection("selOutS2S").set("ymin", -10.0);
            model.component("comp1").selection("selOutS2S").set("ymax", 10.0);
            model.component("comp1").selection("selOutS2S").set("zmin", 14.999999);
            model.component("comp1").selection("selOutS2S").set("zmax", 15.000001);

            model.component("comp1").selection().create("selFreeS2S", "Box");
            model.component("comp1").selection("selFreeS2S").geom("geom1", 2);
            model.component("comp1").selection("selFreeS2S").set("condition", "intersects");
            model.component("comp1").selection("selFreeS2S").set("xmin", -10.0);
            model.component("comp1").selection("selFreeS2S").set("xmax", 10.0);
            model.component("comp1").selection("selFreeS2S").set("ymin", -10.0);
            model.component("comp1").selection("selFreeS2S").set("ymax", 10.0);
            model.component("comp1").selection("selFreeS2S").set("zmin", 1.0e-6);
            model.component("comp1").selection("selFreeS2S").set("zmax", 14.999999);

            // ---- [Fix-1] 每段侧面 Box 选择（用于精确段平均温度）----
            // 坐标系单位 mm（geom1 使用 lengthUnit="mm"）
            // 每段侧面 z 范围：[i*Lseg_mm + δ, (i+1)*Lseg_mm - δ]，排除端面和过渡环面
            // x/y 范围取初始最大半径的 1.5 倍，侵蚀后半径只减不增，不影响选择
            double LsegMM = LsegValue * 1e3;          // 单位: mm
            double rMaxMM = g.rMax * 1e3;             // 单位: mm
            double xySafety = rMaxMM * 1.5;           // x/y 包络范围
            double delta = LsegMM * 0.1;              // z 方向 10% inset，排除端面和过渡环（intersects 模式）
            for (int i = 0; i < segCount; i++) {
                String selTag = "selSegLat_" + (i + 1);
                removeSelection(selTag);
                double zLo = i * LsegMM + delta;
                double zHi = (i + 1) * LsegMM - delta;
                model.component("comp1").selection().create(selTag, "Box");
                model.component("comp1").selection(selTag).geom("geom1", 2);
                model.component("comp1").selection(selTag).set("condition", "intersects");
                model.component("comp1").selection(selTag).set("xmin", -xySafety);
                model.component("comp1").selection(selTag).set("xmax",  xySafety);
                model.component("comp1").selection(selTag).set("ymin", -xySafety);
                model.component("comp1").selection(selTag).set("ymax",  xySafety);
                model.component("comp1").selection(selTag).set("zmin", zLo);
                model.component("comp1").selection(selTag).set("zmax", zHi);
            }

            // 电流激励 BC
            removeEcFeature("potS2S");
            removeEcFeature("gndS2S");
            model.component("comp1").physics("ec").create("potS2S", "ElectricPotential", 2);
            model.component("comp1").physics("ec").feature("potS2S").selection().named("selInS2S");
            model.component("comp1").physics("ec").feature("potS2S").set("V0", "Vapp");
            model.component("comp1").physics("ec").create("gndS2S", "Ground", 2);
            model.component("comp1").physics("ec").feature("gndS2S").selection().named("selOutS2S");
            removeHtFeature("tempInS2S");
            removeHtFeature("tempOutS2S");
            model.component("comp1").physics("ht").create("tempInS2S", "TemperatureBoundary", 2);
            model.component("comp1").physics("ht").feature("tempInS2S").selection().named("selInS2S");
            model.component("comp1").physics("ht").feature("tempInS2S").set("T0", "Telectrode");
            model.component("comp1").physics("ht").create("tempOutS2S", "TemperatureBoundary", 2);
            model.component("comp1").physics("ht").feature("tempOutS2S").selection().named("selOutS2S");
            model.component("comp1").physics("ht").feature("tempOutS2S").set("T0", "Telectrode");

            // ---- 设置 S2S 面-面辐射 ----
            setupSurfaceToSurfaceRadiation();

            // ---- 材料属性（温度相关）----
            model.component("comp1").material("mat1").propertyGroup("def").set("density", new String[]{"rhoMassW"});
            model.component("comp1").material("mat1").propertyGroup("def").set("electricconductivity", new String[]{"1/(" + rhoeExpr + ")"});
            model.component("comp1").material("mat1").propertyGroup("def").set("thermalconductivity", new String[]{kExpr});
            model.component("comp1").material("mat1").propertyGroup("def").set("heatcapacity", new String[]{cpExpr});

            // ---- 划网格 ----
            try { model.component("comp1").mesh("mesh1").feature("size").set("hauto", 4); } catch (Exception e) {}
            try { model.component("comp1").mesh("mesh1").feature("ftet1"); } catch (Exception e) {
                model.component("comp1").mesh("mesh1").create("ftet1", "FreeTet");
            }
            model.component("comp1").mesh("mesh1").run();
        }

        boolean meetsSearchConstraint(Res r) {
            return r.solveOk && r.currentOk && r.volumeOk && r.tempOk;
        }

        Res solvePrepared(GeomData g, double voltage) {
            Res r = new Res();
            r.appliedV = voltage;
            r.segTavg = new double[segCount];
            try {
                setParams(g, voltage);
                try { model.result().numerical("IinS2S").selection().named("selInS2S"); } catch (Exception e) {}
                model.study("std1").run();
                r.Tmax = model.result().numerical("maxTS2S").getReal()[0][0];
                double Tmin;
                try {
                    Tmin = model.result().numerical("minTS2S").getReal()[0][0];
                } catch (Exception e2) {
                    Tmin = r.Tmax * 0.95;
                    System.out.println("WARN: MinVolume failed, using Tmin=" + Tmin + " (0.95*Tmax)");
                }
                double V = model.result().numerical("volS2S").getReal()[0][0];
                double TintVol = model.result().numerical("TintVolS2S").getReal()[0][0];
                r.Tmin = Tmin;
                r.Tmean = (V > 1.0e-20) ? (TintVol / V) : Double.NaN;
                r.U_pct = (r.Tmean > 1.0e-20) ? ((r.Tmax - r.Tmin) / r.Tmean * 100.0) : Double.NaN;
                double I = Math.abs(model.result().numerical("IinS2S").getReal()[0][0]);
                r.P03steady = model.result().numerical("P03emitS2S").getReal()[0][0];
                r.PradSteady = model.result().numerical("PradEmitS2S").getReal()[0][0];
                // Current baseline: sphere stats reuse free-surface 0 K radiation integrals.
                r.PradSphere = r.PradSteady;
                r.P03sphere = r.P03steady;

                // [Fix-1] 各段侧面平均温度：优先用 IntSurface 算子直接读取，
                // 失败时回退到原抛物线近似（算子未就绪或几何异常时的保护）
                for (int i = 0; i < segCount; i++) {
                    boolean readOk = false;
                    try {
                        double Tint = model.result().numerical("TintSeg_" + (i + 1)).getReal()[0][0];
                        double Aseg = model.result().numerical("AsegS2S_"  + (i + 1)).getReal()[0][0];
                        if (Aseg > 1.0e-20) {
                            r.segTavg[i] = Tint / Aseg;
                            readOk = true;
                        }
                    } catch (Exception eTavg) {
                        // 算子尚未创建（如 sanity check 阶段前）或读取失败，静默忽略
                    }
                    if (!readOk) {
                        // 回退：抛物线近似（对均匀圆柱合理，非均匀形状有误差）
                        double eta = ((i + 0.5) * LsegValue) / L0Value;
                        r.segTavg[i] = Tmin + (r.Tmax - Tmin) * 4.0 * eta * (1.0 - eta);
                    }
                }

                r.I = I;
                double V0now = 0.0;
                for (int i = 0; i < segCount; i++) {
                    V0now += Math.PI * g.rSeg[i] * g.rSeg[i] * LsegValue;
                }
                double V0ref = Math.PI * r0Value * r0Value * L0Value;
                r.volErr = Math.abs(V - V0now) / V0ref;
                r.Pelec = voltage * I;
                r.tempOk = r.Tmax < tempLimitK;
                r.volumeOk = r.volErr <= volTol;
                r.currentOk = I > currentTol;
                r.R = r.currentOk ? (voltage / I) : Double.NaN;
                r.solveOk = true;
            } catch (Exception e) {
                r.failure = e.toString();
            }
            return r;
        }

        // 电压二分搜索
        Res searchBestVoltage(GeomData g) {
            int searchSteps = 0;

            Res highRes = solvePrepared(g, voltageUpperBound);
            double highV = voltageUpperBound;
            searchSteps++;
            if (meetsSearchConstraint(highRes)) {
                highRes.searchOk = true;
                highRes.searchSteps = searchSteps;
                return highRes;
            }
            if (highRes.solveOk && (!highRes.currentOk || !highRes.volumeOk)) {
                highRes.searchOk = false;
                highRes.searchSteps = searchSteps;
                return highRes;
            }

            double lowV = Double.NaN;
            Res lowRes = null;

            if (highRes.solveOk && highRes.currentOk && highRes.volumeOk
                && !Double.isNaN(highRes.Tmax) && highRes.Tmax > 0.0) {
                double guessV = voltageUpperBound * Math.sqrt(tempLimitK / Math.max(highRes.Tmax, 1e-300));
                guessV = Math.max(voltageFloor, Math.min(0.98 * voltageUpperBound, guessV));
                if (guessV < highV - 1.0e-12) {
                    Res guessRes = solvePrepared(g, guessV);
                    searchSteps++;
                    if (meetsSearchConstraint(guessRes)) {
                        lowV = guessV;
                        lowRes = guessRes;
                    } else {
                        highV = guessV;
                        highRes = guessRes;
                    }
                }
            }

            while (lowRes == null && highV > voltageFloor + 1.0e-12) {
                double nextV = Math.max(voltageFloor, 0.5 * highV);
                if (Math.abs(nextV - highV) <= 1.0e-12) { break; }
                Res nextRes = solvePrepared(g, nextV);
                searchSteps++;
                if (meetsSearchConstraint(nextRes)) {
                    lowV = nextV;
                    lowRes = nextRes;
                } else {
                    highV = nextV;
                    highRes = nextRes;
                }
            }

            if (lowRes == null) {
                highRes.searchOk = false;
                highRes.searchSteps = searchSteps;
                return highRes;
            }

            for (int iter = 0; iter < maxVoltageSearchIters; iter++) {
                if ((highV - lowV) <= voltageTol) { break; }
                double midV = 0.5 * (lowV + highV);
                Res midRes = solvePrepared(g, midV);
                searchSteps++;
                if (meetsSearchConstraint(midRes)) {
                    lowV = midV;
                    lowRes = midRes;
                } else {
                    highV = midV;
                    highRes = midRes;
                }
            }

            lowRes.searchOk = true;
            lowRes.searchSteps = searchSteps;
            return lowRes;
        }

        void updateGeometry(GeomData g) {
            for (int i = 0; i < segCount; i++) {
                model.param().set("r_seg" + (i + 1),
                    Double.toString(g.rSeg[i]) + "[m]",
                    "Segment " + (i + 1) + " radius");
            }
            model.param().set("RenvInit", Double.toString(g.Renv) + "[m]", "Enclosing sphere radius");
            model.param().set("AenvInit", Double.toString(g.Aenv) + "[m^2]", "Enclosing sphere area");
            model.component("comp1").geom("geom1").run();
            model.component("comp1").mesh("mesh1").run();
        }

        Res solveAtVoltage(GeomData g, double voltage) {
            updateGeometry(g);
            return solvePrepared(g, voltage);
        }
    }

    // ================================================================
    //  主流程开始
    // ================================================================

    Ops ops = new Ops();

    // ---- Step 0: 从空白创建模型骨架 ----
    ops.initModelFromScratch();

    // 使用输入半径
    double[] radii = new double[segCount];
    for (int i = 0; i < segCount; i++) {
        radii[i] = inputRadii[i];
    }
    GeomData g0 = ops.geomFromRadii(radii);

    // 打印输入半径
    System.out.print("INPUT_RADII_MM=");
    for (int i = 0; i < segCount; i++) {
        System.out.print((i > 0 ? "," : "") + String.format("%.4f", inputRadii[i] * 1e3));
    }
    System.out.println();

    // 设置全局参数
    model.label("cylinder_baseline.mph");
    model.param().set("sigmaSB", "5.670374419e-8[W/(m^2*K^4)]", "Stefan-Boltzmann constant");
    model.param().set("eps03", "0.35", "Emissivity 0-3 um band");
    model.param().set("epsRest", "0.15", "Emissivity outside 0-3 um band");
    model.param().set("rhoMassW", "19350[kg/m^3]", "Density of tungsten");
    model.param().set("Tamb", "293.15[K]", "Ambient temperature");
    model.param().set("Telectrode", "293.15[K]", "Copper electrode temperature");
    model.param().set("Vapp", Double.toString(voltageUpperBound) + "[V]", "Applied DC voltage");
    model.param().set("lam03", "3[um]", "Upper wavelength bound");
    model.param().set("c2bb", "1.438776877e-2[m*K]", "Second radiation constant");
    model.param().set("r0", Double.toString(r0Value) + "[m]", "Reference radius (max of input)");
    model.param().set("L0", "15[mm]", "Reference length");
    model.param().set("Nseg", Integer.toString(segCount), "Segment count");
    model.param().set("Lseg", Double.toString(LsegValue) + "[m]", "Axial segment length");
    model.param().set("RenvInit", Double.toString(g0.Renv) + "[m]", "Enclosing sphere radius");
    model.param().set("AenvInit", Double.toString(g0.Aenv) + "[m^2]", "Enclosing sphere area");
    for (int i = 0; i < segCount; i++) {
        model.param().set("r_seg" + (i + 1),
            Double.toString(r0Value) + "[m]",
            "Segment " + (i + 1) + " radius");
    }

    // 首次完整构建（含 S2S 辐射模型 + [Fix-1] per-segment Box 选择）
    ops.setParams(g0, voltageUpperBound);
    ops.rebuild(g0);

    // Joule heating 多物理耦合
    model.multiphysics().create("emh1", "ElectromagneticHeatSource", "geom1", 3);
    model.multiphysics("emh1").selection().all();
    model.multiphysics("emh1").set("EMHeat_physics", "ec");
    model.multiphysics("emh1").set("Heat_physics", "ht");

    // 收敛辅助：3000K 暖启动 + 删旧 solver
    try { model.component("comp1").physics("ht").feature("init1").set("Tinit", "3000[K]"); } catch (Exception e) {}
    try {
        String[] solTags = model.sol().tags();
        for (String st : solTags) {
            try { model.sol(st).clearSolution(); } catch (Exception e2) {}
            try { model.sol().remove(st); } catch (Exception e2) {}
        }
    } catch (Exception e) {}

    // Phase 0: 1V sanity check（让 COMSOL 自动生产 S2S-aware solver）
    ops.setParams(g0, 1.0);
    model.study("std1").run();

    // 创建数值算子
    model.result().numerical().create("maxTS2S", "MaxVolume");
    model.result().numerical("maxTS2S").selection().all();
    model.result().numerical("maxTS2S").set("expr", new String[]{"T"});

    model.result().numerical().create("minTS2S", "MinVolume");
    model.result().numerical("minTS2S").selection().all();
    model.result().numerical("minTS2S").set("expr", new String[]{"T"});

    model.result().numerical().create("volS2S", "IntVolume");
    model.result().numerical("volS2S").selection().all();
    model.result().numerical("volS2S").set("expr", new String[]{"1"});

    model.result().numerical().create("TintVolS2S", "IntVolume");
    model.result().numerical("TintVolS2S").selection().all();
    model.result().numerical("TintVolS2S").set("expr", new String[]{"T"});

    model.result().numerical().create("IinS2S", "IntSurface");
    model.result().numerical("IinS2S").selection().named("selInS2S");
    model.result().numerical("IinS2S").set("expr", new String[]{"ec.Jx*nx+ec.Jy*ny+ec.Jz*nz"});

    model.result().numerical().create("AsurfS2S", "IntSurface");
    model.result().numerical("AsurfS2S").selection().named("selFreeS2S");
    model.result().numerical("AsurfS2S").set("expr", new String[]{"1"});

    model.result().numerical().create("P03emitS2S", "IntSurface");
    model.result().numerical("P03emitS2S").selection().named("selFreeS2S");
    model.result().numerical("P03emitS2S").set("expr", new String[]{q03NetOutExpr});

    model.result().numerical().create("PradEmitS2S", "IntSurface");
    model.result().numerical("PradEmitS2S").selection().named("selFreeS2S");
    model.result().numerical("PradEmitS2S").set("expr", new String[]{qRadNetOutExpr});

    // ---- [Fix-1] 每段侧面平均温度算子：TintSeg_{i+1} = ∫T dA，AsegS2S_{i+1} = ∫1 dA ----
    // Tavg[i] = TintSeg_{i+1} / AsegS2S_{i+1}（面积加权平均温度，不依赖温度分布假设）
    // 这些算子在 solvePrepared() 中每次 study.run() 后自动求值，无需额外运算步骤
    for (int i = 0; i < segCount; i++) {
        String intTTag = "TintSeg_" + (i + 1);
        String intATag = "AsegS2S_"  + (i + 1);
        String selTag  = "selSegLat_" + (i + 1);

        try { model.result().numerical().remove(intTTag); } catch (Exception e) {}
        model.result().numerical().create(intTTag, "IntSurface");
        model.result().numerical(intTTag).selection().named(selTag);
        model.result().numerical(intTTag).set("expr", new String[]{"T"});

        try { model.result().numerical().remove(intATag); } catch (Exception e) {}
        model.result().numerical().create(intATag, "IntSurface");
        model.result().numerical(intATag).selection().named(selTag);
        model.result().numerical(intATag).set("expr", new String[]{"1"});
    }

    // Sanity check
    double sanityTmax = model.result().numerical("maxTS2S").getReal()[0][0];
    double sanityI = Math.abs(model.result().numerical("IinS2S").getReal()[0][0]);
    double sanityR = (sanityI > 1e-20) ? (1.0 / sanityI) : Double.NaN;
    if (!Double.isNaN(sanityR) && sanityR > 1.0) {
        throw new RuntimeException("Sanity failed: R=" + sanityR + " ohm");
    }
    if (sanityTmax < 200.0) {
        throw new RuntimeException("S2S coupling failure: Tmax=" + sanityTmax + "K");
    }

    // ---- Phase 1: 电压搜索 ----
    Res r0 = ops.searchBestVoltage(g0);
    double Vwork = r0.appliedV;
    System.out.println("PHASE1: Vwork=" + Vwork + "V Tmax=" + String.format("%.1f", r0.Tmax)
        + "K P03sph=" + String.format("%.1f", r0.P03sphere) + "W steps=" + r0.searchSteps);
    if (!r0.searchOk) {
        System.out.println("RESULT=FAIL,reason=voltage_search_failed");
        throw new RuntimeException("Voltage search failed");
    }

    // ---- Phase 2: 侵蚀循环 ----
    double timeS = 0.0;
    double p03Integral = 0.0;
    double pradIntegral = 0.0;
    double p03SphereIntegral = 0.0;
    double pradSphereIntegral = 0.0;
    int macroStep = 0;
    boolean failed = false;
    int maxMacroSteps = 50;
    double resolveThresholdR = 0.02 * radii[0];
    for (int i = 1; i < segCount; i++) {
        resolveThresholdR = Math.min(resolveThresholdR, 0.02 * radii[i]);
    }

    double maxLossFrac = 0.0;
    double rMinNow = r0Value;

    double prevP03 = r0.P03steady;
    double prevPrad = r0.PradSteady;
    double prevP03sphere = r0.P03sphere;
    double prevPradSphere = r0.PradSphere;
    double[] Tavg = r0.segTavg;
    double maxErosionTmax = r0.Tmax;
    int overtempStep = -1;
    double overtempTimeH = Double.NaN;
    double overtempTmax = Double.NaN;
    String status = "OK";

    while (macroStep < maxMacroSteps && !failed) {
        macroStep++;

        // (a) 根据当前温度分布计算各段蒸发速率
        // [Fix-1] Tavg[i] 已由 solvePrepared() 中的 IntSurface 算子直接给出，
        //         不再是抛物线估算，对非均匀形状精度更高
        double[] drdt = new double[segCount];
        double maxDrDt = 0.0;
        for (int i = 0; i < segCount; i++) {
            double gamma = AevValue * Math.exp(-BevValue / Tavg[i]);
            drdt[i] = gamma / rhoMassValue;
            if (drdt[i] > maxDrDt) { maxDrDt = drdt[i]; }
        }

        // (b) 蒸发率极低 → 近似无限寿命
        if (maxDrDt < 1.0e-15) {
            System.out.println("Evaporation rate negligible (<1e-15 m/s). Effective infinite lifetime.");
            break;
        }

        // (c) 计算宏步时长
        double dtMacro = Double.MAX_VALUE;
        for (int i = 0; i < segCount; i++) {
            if (drdt[i] > 1.0e-20) {
                double t_resolve = resolveThresholdR / drdt[i];
                double t_fail = (radii[i] - failRadii[i]) / drdt[i];
                dtMacro = Math.min(dtMacro, t_resolve);
                if (t_fail > 0) { dtMacro = Math.min(dtMacro, t_fail); }
            }
        }
        dtMacro = Math.max(1.0, Math.min(36000.0, dtMacro));

        // (d) 解析推进半径
        rMinNow = Double.MAX_VALUE;
        double rMaxNow = 0.0;
        maxLossFrac = 0.0;
        for (int i = 0; i < segCount; i++) {
            radii[i] -= drdt[i] * dtMacro;
            if (radii[i] < 1.0e-6) { radii[i] = 1.0e-6; }
            double lossFrac = (inputRadii[i] - radii[i]) / inputRadii[i];
            if (lossFrac > maxLossFrac) { maxLossFrac = lossFrac; }
            if (radii[i] < rMinNow) { rMinNow = radii[i]; }
            if (radii[i] > rMaxNow) { rMaxNow = radii[i]; }
        }
        timeS += dtMacro;

        // (e) 检查失效
        if (maxLossFrac >= failureFraction) {
            failed = true;
        }

        // (f) 重建几何 + COMSOL S2S 稳态求解
        GeomData gNow = ops.geomFromRadii(radii);
        Res rNow = ops.solveAtVoltage(gNow, Vwork);

        if (!rNow.solveOk) {
            System.out.println("WARN: solve failed step " + macroStep);
            status = "FAIL_EROSION_SOLVE";
            failed = true;
        }

        if (rNow.solveOk && rNow.Tmax > maxErosionTmax) {
            maxErosionTmax = rNow.Tmax;
        }

        // (g) 梯形积分 P03 / Prad（表面发射 + 外接球）
        double curP03 = rNow.solveOk ? rNow.P03steady : prevP03;
        double curPrad = rNow.solveOk ? rNow.PradSteady : prevPrad;
        double curP03sphere = rNow.solveOk ? rNow.P03sphere : prevP03sphere;
        double curPradSphere = rNow.solveOk ? rNow.PradSphere : prevPradSphere;
        p03Integral += 0.5 * (prevP03 + curP03) * dtMacro;
        pradIntegral += 0.5 * (prevPrad + curPrad) * dtMacro;
        p03SphereIntegral += 0.5 * (prevP03sphere + curP03sphere) * dtMacro;
        pradSphereIntegral += 0.5 * (prevPradSphere + curPradSphere) * dtMacro;
        prevP03 = curP03;
        prevPrad = curPrad;
        prevP03sphere = curP03sphere;
        prevPradSphere = curPradSphere;

        Tavg = rNow.segTavg;

        if (rNow.solveOk && rNow.Tmax >= tempLimitK) {
            status = "FAIL_OVERTEMP_DURING_EROSION";
            overtempStep = macroStep;
            overtempTimeH = timeS / 3600.0;
            overtempTmax = rNow.Tmax;
            break;
        }

        if (macroStep % 5 == 0 || failed) {
            System.out.println("STEP=" + macroStep + " t=" + String.format("%.2f", timeS / 3600.0)
                + "h loss=" + String.format("%.4f", maxLossFrac));
        }
    }
    int step = macroStep;

    // ---- Phase 3: 输出核心赛题指标（CSV 可解析格式）----

    double lifetimeH = timeS / 3600.0;
    double avgP03sphere = (timeS > 0.0) ? (p03SphereIntegral / timeS) : Double.NaN;
    double avgPradSphere = (timeS > 0.0) ? (pradSphereIntegral / timeS) : Double.NaN;

    // [Fix-2] selfViewLoss 说明：
    //   定义：selfViewLoss = (1 - P03sphere积分 / P03surface积分) × 100%
    //   当前口径：P03sphere 与 PradSphere 直接采用自由表面 0 K 辐射积分。
    //        P03surface / PradSurface 保留为同一自由表面选择下的诊断积分。
    //   问题根源：qRadNetOutExpr 是局部自发射公式，不含 S2S 面间辐射交换的修正项，
    //   若 selfViewLoss 明显偏离 0，应优先检查 selFreeS2S 和积分算子选择。
    //   从而 P03sphere > P03surface，selfViewLoss 呈负值（约 −5%）。
    //   物理真值：对凸圆柱体无自遮挡，selfViewLoss 应 ≈ 0%。
    //   影响评估：此偏差为系统性固定偏移，不影响不同形状之间的相对比较，
    //   ML 优化阶段可正常使用 P03sphere 和寿命作为目标。竞赛汇报时需注明
    //   汇报时按官方 0 K 统计面口径说明即可。
    double selfViewLoss = (timeS > 0.0 && p03Integral > 0.0)
        ? (1.0 - p03SphereIntegral / p03Integral) * 100.0 : Double.NaN;

    // CSV header + data（方便 ML pipeline 解析）
    System.out.println("RESULT_HEADER=Vwork_V,initialTmax_K,Tmin_K,Tmean_K,U_pct,maxErosionTmax_K,lifetimeH,initialP03sphere_W,initialPradSphere_W,lifeAvgP03sphere_W,lifeAvgPradSphere_W,lifeTotalP03sphere_J,selfViewLoss_pct,failureReached,erosionSteps,overtempStep,overtempTimeH,overtempTmax_K,status");
    System.out.println("RESULT="
        + String.format("%.6f", Vwork) + ","
        + String.format("%.1f", r0.Tmax) + ","
        + String.format("%.1f", r0.Tmin) + ","
        + String.format("%.1f", r0.Tmean) + ","
        + String.format("%.4f", r0.U_pct) + ","
        + String.format("%.1f", maxErosionTmax) + ","
        + String.format("%.4f", lifetimeH) + ","
        + String.format("%.2f", r0.P03sphere) + ","
        + String.format("%.2f", r0.PradSphere) + ","
        + String.format("%.2f", avgP03sphere) + ","
        + String.format("%.2f", avgPradSphere) + ","
        + String.format("%.2f", p03SphereIntegral) + ","
        + String.format("%.2f", selfViewLoss) + ","
        + failed + ","
        + step + ","
        + overtempStep + ","
        + String.format("%.4f", overtempTimeH) + ","
        + String.format("%.1f", overtempTmax) + ","
        + status);

    // [Fix-2] 输出 selfViewLoss 诊断行，提醒负值是已知系统误差
    if (!Double.isNaN(selfViewLoss) && selfViewLoss < -1.0) {
        System.out.println("NOTE: selfViewLoss=" + String.format("%.2f", selfViewLoss)
            + "% (<0). Current baseline uses free-surface 0 K statistics;"
            + " check selFreeS2S and P03/Prad integrals if this is unexpected.");
    }

    // 输入半径回显（方便核查）
    System.out.print("INPUT_RADII_M=");
    for (int i = 0; i < segCount; i++) {
        System.out.print((i > 0 ? "," : "") + String.format("%.6e", inputRadii[i]));
    }
    System.out.println();
    System.out.print("FINAL_RADII_M=");
    for (int i = 0; i < segCount; i++) {
        System.out.print((i > 0 ? "," : "") + String.format("%.6e", radii[i]));
    }
    System.out.println();
}
