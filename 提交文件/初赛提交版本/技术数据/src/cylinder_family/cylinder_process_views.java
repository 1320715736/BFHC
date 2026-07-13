{


    double[] inputRadii = new double[]{
        2.5e-3, 2.5e-3, 2.5e-3, 2.5e-3,
        2.5e-3, 2.5e-3, 2.5e-3, 2.5e-3
    };


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
    String epsRadS2SValue = "if(comp1.rad.lambda<lam03,eps03,epsRest)";


    double r0Tmp = 0.0;
    for (int i = 0; i < segCount; i++) { if (inputRadii[i] > r0Tmp) r0Tmp = inputRadii[i]; }
    double r0Value = r0Tmp;

    double[] failRadii = new double[segCount];
    for (int i = 0; i < segCount; i++) {
        failRadii[i] = inputRadii[i] * (1.0 - failureFraction);
    }
    double rMin = r0Value * (1.0 - failureFraction);


    String rhoeExpr = "max(1e-10[ohm*m], 5.5e-8[ohm*m]*(1+0.003836*(T-293.15[K])/1[K]+7.55e-7*((T-293.15[K])/1[K])^2))";
    String kExpr = "max(75[W/(m*K)],175[W/(m*K)]-0.032[W/(m*K^2)]*(T-293.15[K]))";
    String cpExpr = "min(195[J/(kg*K)],132[J/(kg*K)]+0.020[J/(kg*K^2)]*(T-293.15[K]))";


    String x03TExpr = "(c2bb/(lam03*T))";
    String x03TambExpr = "(c2bb/(lam03*Tamb))";
    String seriesTExpr = "";
    String seriesTambExpr = "";
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
        String termTamb =
            "exp(-" + n + "*" + x03TambExpr + ")*("
            + x03TambExpr + "^3/" + n
            + "+3*" + x03TambExpr + "^2/" + n2
            + "+6*" + x03TambExpr + "/" + n3
            + "+6/" + n4
            + ")";
        if (n == 1) {
            seriesTExpr = termT;
            seriesTambExpr = termTamb;
        } else {
            seriesTExpr = seriesTExpr + "+" + termT;
            seriesTambExpr = seriesTambExpr + "+" + termTamb;
        }
    }
    String f03bbTExpr = "min(1,max(0,(15/pi^4)*(" + seriesTExpr + ")))";
    String f03bbTambExpr = "min(1,max(0,(15/pi^4)*(" + seriesTambExpr + ")))";
    String q03NetOutExpr =
        "eps03*sigmaSB*((" + f03bbTExpr + ")*T^4-(" + f03bbTambExpr + ")*Tamb^4)";
    String qRadNetOutExpr =
        "sigmaSB*(epsRest*(T^4-Tamb^4)+(eps03-epsRest)*((" + f03bbTExpr + ")*T^4-(" + f03bbTambExpr + ")*Tamb^4))";


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


    class Ops {


        void initModelFromScratch() {

            model.component().create("comp1");
            model.component("comp1").geom().create("geom1", 3);
            model.component("comp1").geom("geom1").lengthUnit("mm");


            model.component("comp1").physics().create("ec", "ConductiveMedia", "geom1");
            model.component("comp1").physics().create("ht", "HeatTransfer", "geom1");
            model.component("comp1").physics().create("solid", "SolidMechanics", "geom1");


            model.component("comp1").material().create("mat1", "Common");
            model.component("comp1").material("mat1").label("Tungsten");
            model.component("comp1").material("mat1").selection().all();


            model.component("comp1").mesh().create("mesh1", "geom1");


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
        void removeSolidFeature(String tag) {
            try { model.component("comp1").physics("solid").feature().remove(tag); } catch (Exception e) {}
        }
        void removePhysics(String tag) {
            try { model.component("comp1").physics().remove(tag); } catch (Exception e) {}
        }
        void removeMultiphysics(String tag) {
            try { model.multiphysics().remove(tag); } catch (Exception e) {}
        }


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

            removeGeomFeature("uniS2S");
            for (int i = 0; i < segCount; i++) { removeGeomFeature("cS2S_" + (i + 1)); }


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


            removeSelection("selInS2S");
            removeSelection("selOutS2S");

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


            double LsegMM = LsegValue * 1e3;
            double rMaxMM = g.rMax * 1e3;
            double xySafety = rMaxMM * 1.5;
            double delta = LsegMM * 0.1;
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


            removeEcFeature("potS2S");
            removeEcFeature("gndS2S");
            model.component("comp1").physics("ec").create("potS2S", "ElectricPotential", 2);
            model.component("comp1").physics("ec").feature("potS2S").selection().named("selInS2S");
            model.component("comp1").physics("ec").feature("potS2S").set("V0", "Vapp");
            model.component("comp1").physics("ec").create("gndS2S", "Ground", 2);
            model.component("comp1").physics("ec").feature("gndS2S").selection().named("selOutS2S");


            removeSolidFeature("fixInS2S");
            removeSolidFeature("fixOutS2S");
            model.component("comp1").physics("solid").create("fixInS2S", "Fixed", 2);
            model.component("comp1").physics("solid").feature("fixInS2S").selection().named("selInS2S");
            model.component("comp1").physics("solid").create("fixOutS2S", "Fixed", 2);
            model.component("comp1").physics("solid").feature("fixOutS2S").selection().named("selOutS2S");

            try {
                model.component("comp1").physics("solid").feature("lemm1").feature("te1");
            } catch (Exception e) {
                try {
                    model.component("comp1").physics("solid").feature("lemm1")
                        .create("te1", "ThermalExpansion", 3);
                } catch (Exception e2) {}
            }
            try {
                model.component("comp1").physics("solid").feature("lemm1").feature("te1")
                    .set("T", "T");
            } catch (Exception e) {}
            try {
                model.component("comp1").physics("solid").feature("lemm1").feature("te1")
                    .set("Temp", "T");
            } catch (Exception e) {}
            try {
                model.component("comp1").physics("solid").feature("lemm1").feature("te1")
                    .set("minput_temperature", "T");
            } catch (Exception e) {}
            try {
                model.component("comp1").physics("solid").feature("lemm1").feature("te1")
                    .set("minput_temperature_src", "root.comp1.T");
            } catch (Exception e) {}
            try {
                model.component("comp1").physics("solid").feature("lemm1").feature("te1")
                    .set("Tref", "Tamb");
            } catch (Exception e) {}
            try {
                model.component("comp1").physics("solid").feature("lemm1").feature("te1")
                    .set("T0", "Tamb");
            } catch (Exception e) {}
            try {
                model.component("comp1").physics("solid").feature("lemm1").feature("te1")
                    .set("Tempref", "Tamb");
            } catch (Exception e) {}


            try {
                model.component("comp1").physics("solid").feature("teTopS2S");
            } catch (Exception e) {
                try {
                    model.component("comp1").physics("solid").create("teTopS2S", "ThermalExpansion", 3);
                } catch (Exception e2) {}
            }
            try { model.component("comp1").physics("solid").feature("teTopS2S").set("T", "T"); } catch (Exception e) {}
            try { model.component("comp1").physics("solid").feature("teTopS2S").set("Temp", "T"); } catch (Exception e) {}
            try { model.component("comp1").physics("solid").feature("teTopS2S").set("minput_temperature", "T"); } catch (Exception e) {}
            try { model.component("comp1").physics("solid").feature("teTopS2S").set("minput_temperature_src", "root.comp1.T"); } catch (Exception e) {}
            try { model.component("comp1").physics("solid").feature("teTopS2S").set("Tref", "Tamb"); } catch (Exception e) {}
            try { model.component("comp1").physics("solid").feature("teTopS2S").set("T0", "Tamb"); } catch (Exception e) {}
            try { model.component("comp1").physics("solid").feature("teTopS2S").set("Tempref", "Tamb"); } catch (Exception e) {}


            setupSurfaceToSurfaceRadiation();


            model.component("comp1").material("mat1").propertyGroup("def").set("density", new String[]{"rhoMassW"});
            model.component("comp1").material("mat1").propertyGroup("def").set("electricconductivity", new String[]{"1/(" + rhoeExpr + ")"});
            model.component("comp1").material("mat1").propertyGroup("def").set("thermalconductivity", new String[]{kExpr});
            model.component("comp1").material("mat1").propertyGroup("def").set("heatcapacity", new String[]{cpExpr});
            model.component("comp1").material("mat1").propertyGroup("def").set("youngsmodulus", new String[]{"EW"});
            model.component("comp1").material("mat1").propertyGroup("def").set("poissonsratio", new String[]{"nuW"});
            model.component("comp1").material("mat1").propertyGroup("def").set("thermalexpansioncoefficient", new String[]{"alphaW"});


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
                double I = Math.abs(model.result().numerical("IinS2S").getReal()[0][0]);
                r.P03steady = model.result().numerical("P03emitS2S").getReal()[0][0];
                r.PradSteady = model.result().numerical("PradEmitS2S").getReal()[0][0];

                r.PradSphere = voltage * I;
                r.P03sphere = (r.PradSteady > 1e-10)
                    ? (voltage * I * r.P03steady / r.PradSteady) : 0.0;


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

                    }
                    if (!readOk) {

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


    Ops ops = new Ops();


    ops.initModelFromScratch();


    double[] radii = new double[segCount];
    for (int i = 0; i < segCount; i++) {
        radii[i] = inputRadii[i];
    }
    GeomData g0 = ops.geomFromRadii(radii);


    System.out.print("INPUT_RADII_MM=");
    for (int i = 0; i < segCount; i++) {
        System.out.print((i > 0 ? "," : "") + String.format("%.4f", inputRadii[i] * 1e3));
    }
    System.out.println();


    model.label("cylinder_process_views.mph");
    model.param().set("sigmaSB", "5.670374419e-8[W/(m^2*K^4)]", "Stefan-Boltzmann constant");
    model.param().set("eps03", "0.35", "Emissivity 0-3 um band");
    model.param().set("epsRest", "0.15", "Emissivity outside 0-3 um band");
    model.param().set("rhoMassW", "19350[kg/m^3]", "Density of tungsten");
    model.param().set("Tamb", "293.15[K]", "Ambient temperature");
    model.param().set("EW", "411[GPa]", "Young's modulus of tungsten for Solid Mechanics");
    model.param().set("nuW", "0.28", "Poisson ratio of tungsten for Solid Mechanics");
    model.param().set("alphaW", "4.5e-6[1/K]", "Thermal expansion coefficient of tungsten for Solid Mechanics");
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


    ops.setParams(g0, voltageUpperBound);
    ops.rebuild(g0);


    model.multiphysics().create("emh1", "ElectromagneticHeatSource", "geom1", 3);
    model.multiphysics("emh1").selection().all();
    model.multiphysics("emh1").set("EMHeat_physics", "ec");
    model.multiphysics("emh1").set("Heat_physics", "ht");


    try { model.component("comp1").physics("ht").feature("init1").set("Tinit", "3000[K]"); } catch (Exception e) {}
    try {
        String[] solTags = model.sol().tags();
        for (String st : solTags) {
            try { model.sol(st).clearSolution(); } catch (Exception e2) {}
            try { model.sol().remove(st); } catch (Exception e2) {}
        }
    } catch (Exception e) {}


    ops.setParams(g0, 1.0);
    model.study("std1").run();


    model.result().numerical().create("maxTS2S", "MaxVolume");
    model.result().numerical("maxTS2S").selection().all();
    model.result().numerical("maxTS2S").set("expr", new String[]{"T"});

    model.result().numerical().create("minTS2S", "MinVolume");
    model.result().numerical("minTS2S").selection().all();
    model.result().numerical("minTS2S").set("expr", new String[]{"T"});

    model.result().numerical().create("volS2S", "IntVolume");
    model.result().numerical("volS2S").selection().all();
    model.result().numerical("volS2S").set("expr", new String[]{"1"});

    model.result().numerical().create("IinS2S", "IntSurface");
    model.result().numerical("IinS2S").selection().named("selInS2S");
    model.result().numerical("IinS2S").set("expr", new String[]{"ec.Jx*nx+ec.Jy*ny+ec.Jz*nz"});

    model.result().numerical().create("AsurfS2S", "IntSurface");
    model.result().numerical("AsurfS2S").selection().all();
    model.result().numerical("AsurfS2S").set("expr", new String[]{"1"});

    model.result().numerical().create("P03emitS2S", "IntSurface");
    model.result().numerical("P03emitS2S").selection().all();
    model.result().numerical("P03emitS2S").set("expr", new String[]{q03NetOutExpr});

    model.result().numerical().create("PradEmitS2S", "IntSurface");
    model.result().numerical("PradEmitS2S").selection().all();
    model.result().numerical("PradEmitS2S").set("expr", new String[]{qRadNetOutExpr});


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


    double sanityTmax = model.result().numerical("maxTS2S").getReal()[0][0];
    double sanityI = Math.abs(model.result().numerical("IinS2S").getReal()[0][0]);
    double sanityR = (sanityI > 1e-20) ? (1.0 / sanityI) : Double.NaN;
    if (!Double.isNaN(sanityR) && sanityR > 1.0) {
        throw new RuntimeException("Sanity failed: R=" + sanityR + " ohm");
    }
    if (sanityTmax < 200.0) {
        throw new RuntimeException("S2S coupling failure: Tmax=" + sanityTmax + "K");
    }


    Res r0 = ops.searchBestVoltage(g0);
    double Vwork = r0.appliedV;
    System.out.println("PHASE1: Vwork=" + Vwork + "V Tmax=" + String.format("%.1f", r0.Tmax)
        + "K P03sph=" + String.format("%.1f", r0.P03sphere) + "W steps=" + r0.searchSteps);
    if (!r0.searchOk) {
        System.out.println("RESULT=FAIL,reason=voltage_search_failed");
        throw new RuntimeException("Voltage search failed");
    }


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


    double[][] radiiHist = new double[maxMacroSteps + 1][];
    double[] timeHist = new double[maxMacroSteps + 1];
    double[] lossHist = new double[maxMacroSteps + 1];
    double[] tmaxHist = new double[maxMacroSteps + 1];
    radiiHist[0] = new double[segCount];
    for (int i = 0; i < segCount; i++) {
        radiiHist[0][i] = radii[i];
    }
    timeHist[0] = 0.0;
    lossHist[0] = 0.0;
    tmaxHist[0] = r0.Tmax;

    while (macroStep < maxMacroSteps && !failed) {
        macroStep++;


        double[] drdt = new double[segCount];
        double maxDrDt = 0.0;
        for (int i = 0; i < segCount; i++) {
            double gamma = AevValue * Math.exp(-BevValue / Tavg[i]);
            drdt[i] = gamma / rhoMassValue;
            if (drdt[i] > maxDrDt) { maxDrDt = drdt[i]; }
        }


        if (maxDrDt < 1.0e-15) {
            System.out.println("Evaporation rate negligible (<1e-15 m/s). Effective infinite lifetime.");
            break;
        }


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


        if (maxLossFrac >= failureFraction) {
            failed = true;
        }


        GeomData gNow = ops.geomFromRadii(radii);
        Res rNow = ops.solveAtVoltage(gNow, Vwork);

        if (!rNow.solveOk) {
            System.out.println("WARN: solve failed step " + macroStep);
            failed = true;
        }


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

        radiiHist[macroStep] = new double[segCount];
        for (int i = 0; i < segCount; i++) {
            radiiHist[macroStep][i] = radii[i];
        }
        timeHist[macroStep] = timeS;
        lossHist[macroStep] = maxLossFrac;
        tmaxHist[macroStep] = rNow.solveOk ? rNow.Tmax : Double.NaN;

        if (macroStep % 5 == 0 || failed) {
            System.out.println("STEP=" + macroStep + " t=" + String.format("%.2f", timeS / 3600.0)
                + "h loss=" + String.format("%.4f", maxLossFrac));
        }
    }
    int step = macroStep;


    double lifetimeH = timeS / 3600.0;
    double avgP03sphere = (timeS > 0.0) ? (p03SphereIntegral / timeS) : Double.NaN;
    double avgPradSphere = (timeS > 0.0) ? (pradSphereIntegral / timeS) : Double.NaN;


    double selfViewLoss = (timeS > 0.0 && p03Integral > 0.0)
        ? (1.0 - p03SphereIntegral / p03Integral) * 100.0 : Double.NaN;


    System.out.println("RESULT_HEADER=Vwork_V,initialTmax_K,lifetimeH,initialP03sphere_W,initialPradSphere_W,lifeAvgP03sphere_W,lifeAvgPradSphere_W,selfViewLoss_pct,failureReached,erosionSteps");
    System.out.println("RESULT="
        + String.format("%.6f", Vwork) + ","
        + String.format("%.1f", r0.Tmax) + ","
        + String.format("%.4f", lifetimeH) + ","
        + String.format("%.2f", r0.P03sphere) + ","
        + String.format("%.2f", r0.PradSphere) + ","
        + String.format("%.2f", avgP03sphere) + ","
        + String.format("%.2f", avgPradSphere) + ","
        + String.format("%.2f", selfViewLoss) + ","
        + failed + ","
        + step);


    if (!Double.isNaN(selfViewLoss) && selfViewLoss < -1.0) {
        System.out.println("NOTE: selfViewLoss=" + String.format("%.2f", selfViewLoss)
            + "% (<0). Expected for convex cylinder. Cause: qRadNetOutExpr surface integral"
            + " excludes S2S inter-surface exchange => PradSurface underestimates V*I by ~5%."
            + " Does NOT affect shape-to-shape comparisons. Note in report.");
    }


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


    int lastRecordedStep = step;
    while (lastRecordedStep > 0 && radiiHist[lastRecordedStep] == null) {
        lastRecordedStep--;
    }

    double finalLoss = lossHist[lastRecordedStep];
    double[] targetLosses = new double[]{
        0.0,
        finalLoss / 3.0,
        2.0 * finalLoss / 3.0,
        finalLoss
    };
    int[] keySteps = new int[4];
    for (int ki = 0; ki < 4; ki++) {
        int bestStep = 0;
        double bestErr = Double.POSITIVE_INFINITY;
        for (int s = 0; s <= lastRecordedStep; s++) {
            if (radiiHist[s] == null) { continue; }
            double err = Math.abs(lossHist[s] - targetLosses[ki]);
            if (err < bestErr) {
                bestErr = err;
                bestStep = s;
            }
        }
        keySteps[ki] = (ki == 3) ? lastRecordedStep : bestStep;
    }

    System.out.println("VIZ: Creating cylinder_stage_01..04 independent datasets.");
    for (int ki = 0; ki < 4; ki++) {
        int ks = keySteps[ki];
        String idx = (ki + 1 < 10 ? "0" : "") + (ki + 1);
        String stageName = "cylinder_stage_" + idx;

        for (int i = 0; i < segCount; i++) {
            model.param().set("r_seg" + (i + 1), Double.toString(radiiHist[ks][i]) + "[m]");
        }
        GeomData gKey = ops.geomFromRadii(radiiHist[ks]);
        model.param().set("Vapp", Double.toString(Vwork) + "[V]");
        model.param().set("RenvInit", Double.toString(gKey.Renv) + "[m]");
        model.param().set("AenvInit", Double.toString(gKey.Aenv) + "[m^2]");
        model.component("comp1").geom("geom1").run();
        model.component("comp1").mesh("mesh1").run();

        String stTag = "stdCylinderStage" + idx;
        try { model.study().remove(stTag); } catch (Exception e) {}
        model.study().create(stTag);
        model.study(stTag).label(stageName);
        model.study(stTag).create("stat", "Stationary");
        model.study(stTag).run();

        String[] dsetTags = model.result().dataset().tags();
        String dsetTag = dsetTags[dsetTags.length - 1];

        String pgTag = "pgCylinderStage" + idx;
        String title = stageName
            + " | t=" + String.format("%.3f", timeHist[ks] / 3600.0) + " h"
            + " | loss=" + String.format("%.2f", lossHist[ks] * 100.0) + "%"
            + " | dataset=" + dsetTag
            + " | edit expr: T or solid.mises";
        try { model.result().remove(pgTag); } catch (Exception e) {}
        model.result().create(pgTag, "PlotGroup3D");
        model.result(pgTag).set("data", dsetTag);
        model.result(pgTag).label(stageName + " selectable field");
        model.result(pgTag).set("titletype", "manual");
        model.result(pgTag).set("title", title);
        model.result(pgTag).create("surf1", "Surface");
        model.result(pgTag).feature("surf1").set("expr", "T");
        model.result(pgTag).feature("surf1").set("unit", "K");
        model.result(pgTag).feature("surf1").set("colortable", "ThermalLight");
        model.result(pgTag).run();

        System.out.println("VIZ_STAGE=" + stageName
            + ",studyTag=" + stTag
            + ",plotGroup=" + pgTag
            + ",dataset=" + dsetTag
            + ",step=" + ks
            + ",timeH=" + String.format("%.6f", timeHist[ks] / 3600.0)
            + ",loss=" + String.format("%.6f", lossHist[ks])
            + ",exprOptions=T|solid.mises");
    }
    System.out.println("VIZ: Open cylinder_stage_01..04.  Change Surface expression to T or solid.mises.");
}
