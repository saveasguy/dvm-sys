#include "pragmas.h"

#include <cassert>

#include "file_ctx.h"

namespace cdvmh {

// DistribAxisRule

bool DistribAxisRule::isConstant() const {
    if (distrType == dtReplicated || distrType == dtBlock)
        return true;
    if (distrType == dtMultBlock && isNumber(multBlockValue.strExpr))
        return true;
    return false;
}

// DistribRule

bool DistribRule::isConstant() const {
    for (int i = 0; i < rank; i++)
        if (!axes[i].isConstant())
            return false;
    return true;
}

// ClauseReduction

std::string ClauseReduction::guessRedType(std::string tokStr) {
    if (!redTypesInitialized)
        initRedTypes();
    std::map<std::string, std::string>::iterator it = redTypes.find(tokStr);
    if (it == redTypes.end())
        return "";
    else
        return it->second;
}

std::string ClauseReduction::toOpenMP() const {
    if (redType == "rf_SUM") return "+";
    else if (redType == "rf_PROD" || redType == "rf_MULT") return "*";
    else if (redType == "rf_AND") return "&";
    else if (redType == "rf_OR") return "|";
    else if (redType == "rf_XOR") return "^";
    else return "";
}

std::pair<std::string, bool> ClauseReduction::toCUDA() const {
    if (redType == "rf_SUM") return std::make_pair(std::string("+"), false);
    if (redType == "rf_PROD" || redType == "rf_MULT") return std::make_pair(std::string("*"), false);
    if (redType == "rf_AND") return std::make_pair(std::string("&"), false);
    if (redType == "rf_OR") return std::make_pair(std::string("|"), false);
    if (redType == "rf_XOR") return std::make_pair("^", false);
    if (redType == "rf_MAX" || redType == "rf_MAXLOC") return std::make_pair(std::string("max"), true);
    if (redType == "rf_MIN" || redType == "rf_MINLOC") return std::make_pair(std::string("min"), true);
    return std::make_pair(std::string(), false);
}

std::string ClauseReduction::toCUDABlock() const {
    if (redType == "rf_SUM") return "__dvmh_blockReduceSum";
    if (redType == "rf_PROD" || redType == "rf_MULT") return "__dvmh_blockReduceProd";
    if (redType == "rf_AND") return "__dvmh_blockReduceAND";
    if (redType == "rf_OR") return "__dvmh_blockReduceOR";
    if (redType == "rf_XOR") return "__dvmh_blcokReduceXOR";
    if (redType == "rf_MAX") return "__dvmh_blockReduceMax";
    if (redType == "rf_MAXLOC") return "__dvmh_blockReduceMaxLoc";
    if (redType == "rf_MIN") return "__dvmh_blockReduceMin";
    if (redType == "rf_MINLOC") return "__dvmh_blockReduceMinLoc";
    return "";
}

std::string ClauseReduction::toClause() const {
    std::string redFunc = redType;
    for (std::map<std::string, std::string>::iterator it = redTypes.begin(); it != redTypes.end(); it++)
        if (redType == it->second)
            redFunc = it->first;
    return redFunc + "(" + arrayName + (isLoc() ? ", " + locName + ", " + locSize.strExpr : "") + ")";
}

void ClauseReduction::initRedTypes() {
    redTypes["sum"] = "rf_SUM";
    redTypes["product"] = "rf_PROD";
    redTypes["max"] = "rf_MAX";
    redTypes["min"] = "rf_MIN";
    redTypes["and"] = "rf_AND";
    redTypes["or"] = "rf_OR";
    redTypes["xor"] = "rf_XOR";
    redTypes["maxloc"] = "rf_MAXLOC";
    redTypes["minloc"] = "rf_MINLOC";
    redTypesInitialized = true;
}

std::map<std::string, std::string> ClauseReduction::redTypes;
bool ClauseReduction::redTypesInitialized = false;

// ClauseAcross

int ClauseAcross::getDepCount() const {
    int res = 0;
    for (int i = 0; i < rank; i++) {
        bool noDep = isNumber(widths[i].first.strExpr) && toNumber(widths[i].first.strExpr) == 0;
        noDep = noDep && isNumber(widths[i].second.strExpr) && toNumber(widths[i].second.strExpr) == 0;
        if (!noDep)
            res++;
    }
    return res;
}

// ClauseRemoteAccess

bool ClauseRemoteAccess::matches(std::string seenExpr, int idx) const {
    assert(idx >= 0 && idx < rank && "Invalid number of axis");
    if (axisRules[idx].axisNumber == -1)
        return true;
    std::string origExpr = axisRules[idx].origExpr.strExpr;
    int i1, i2;
    i1 = 0;
    i2 = 0;
    while (i1 < (int)seenExpr.size() && i2 < (int)origExpr.size()) {
        while (i1 < (int)seenExpr.size() && seenExpr[i1] == ' ')
            i1++;
        while (i2 < (int)origExpr.size() && origExpr[i2] == ' ')
            i2++;
        if ((i1 < (int)seenExpr.size()) != (i2 < (int)origExpr.size()))
            return false;
        if (i1 < (int)seenExpr.size() && i2 < (int)origExpr.size()) {
            if (seenExpr[i1] != origExpr[i2])
                return false;
            i1++;
            i2++;
        }
    }
    bool res = (i1 >= (int)seenExpr.size() && i2 >= (int)origExpr.size());
    cdvmh_log(TRACE, "Matches: seenExpr=%s, origExpr=%s, result=%d", seenExpr.c_str(), origExpr.c_str(), (int)res);
    return res;
}

// PragmaRegion

std::string PragmaRegion::genStrTargets(int targets) {
    const char *strs[] = {"DEVICE_TYPE_HOST", "DEVICE_TYPE_CUDA"};
    return flagsToStr(targets, strs, 2);
}

std::string PragmaRegion::genStrIntent(int intent) {
    const char *strs[] = {"INTENT_IN", "INTENT_OUT", "INTENT_LOCAL", "INTENT_USE"};
    return flagsToStr(intent, strs, 4);
}

std::string PragmaRegion::getFlagsStr() {
    const char *strs[] = {"REGION_ASYNC", "REGION_COMPARE_DEBUG"};
    return flagsToStr(flags, strs, 2);
}

}
