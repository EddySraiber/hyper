# 📋 **Manual QA Testing Guide for Algotrading Agent**

As a QA manual tester, here's your comprehensive testing framework:

## **🎯 Testing Philosophy**
- **Test from user perspective** - What would a real trader expect?
- **Break things intentionally** - Try to find edge cases
- **Document everything** - Screenshots, steps, expected vs actual results
- **Test iteratively** - Retest after fixes

## **📝 QA Testing Checklist**

### **Phase 1: System Availability Tests**
```
□ Dashboard loads at http://localhost:8080/dashboard
□ All UI sections visible (Portfolio, News, Decisions, Logs, Settings)
□ Health endpoint responds at http://localhost:8080/health
□ No browser console errors (press F12 → Console tab)
□ Dashboard auto-refreshes every 5 seconds
```

### **Phase 2: Data Consistency Tests**
```
□ Portfolio Total Value = $100,000 (consistent on refresh)
□ Available Cash = $100,000 (when no positions)
□ Daily P&L = $0.00 (fresh start)
□ Position Count = 0 (no trades yet)
□ Risk Usage = 0.0% (no exposure)
```

### **Phase 3: Real-Time Data Flow Tests**
```
□ Live Logs section shows recent activity
□ News section populates with items (may take 5 mins)
□ News items have titles, sources, and scores
□ Trading Decisions section (likely empty initially)
□ Timestamps update in real-time
```

### **Phase 4: Configuration Tests**
```
□ Change Min Confidence from 0.6 to 0.4 → Click Update
□ Verify success message appears
□ Change Stop Loss % from 0.05 to 0.03 → Click Update
□ Change Take Profit % from 0.10 to 0.15 → Click Update
□ Refresh page - verify settings persist (if implemented)
```

### **Phase 5: Error Handling Tests**
```
□ Enter invalid values in config (negative numbers, text)
□ Try accessing http://localhost:8080/nonexistent
□ Disconnect internet briefly - verify system handles gracefully
□ Wait 10+ minutes - verify system continues operating
```

### **Phase 6: Browser Compatibility**
```
□ Chrome/Chromium
□ Firefox
□ Safari (if on Mac)
□ Mobile view (responsive design)
```

## **🔧 Testing Tools & Techniques**

### **Browser Developer Tools**
```bash
# Open with F12 or right-click → Inspect
1. Console tab - Check for JavaScript errors
2. Network tab - Monitor API calls
3. Application tab - Check for data storage
```

### **Manual API Testing**
```bash
# Test in terminal or browser
curl http://localhost:8080/health
curl http://localhost:8080/api/portfolio
curl http://localhost:8080/api/news
curl http://localhost:8080/api/decisions
```

### **System Monitoring**
```bash
# Monitor logs
docker-compose logs -f algotrading-agent

# Check container health
docker-compose ps
```

## **📊 Test Case Template**

For each test, document:

```markdown
### Test Case: [Test Name]
**Preconditions:** [System state before test]
**Steps:**
1. Step 1
2. Step 2
3. Step 3

**Expected Result:** [What should happen]
**Actual Result:** [What actually happened]
**Status:** ✅ PASS / ❌ FAIL
**Screenshots:** [If applicable]
**Notes:** [Additional observations]
```

## **🚨 Critical Issues to Watch For**

### **High Priority Bugs:**
- Dashboard not loading (HTTP 500/404)
- Portfolio values jumping around
- Configuration updates not working
- System crashes or containers stopping

### **Medium Priority Issues:**
- Slow loading (>10 seconds)
- UI layout problems
- Missing data in news/decisions
- Console warnings

### **Low Priority Issues:**
- Cosmetic problems
- Minor text issues
- Non-critical feature gaps

## **🎯 Specific Test Scenarios**

### **Scenario A: First-Time User**
```
1. Access dashboard for first time
2. Verify all default values are correct
3. Check that help/instructions are clear
4. Test configuration changes
5. Monitor for 15+ minutes
```

### **Scenario B: Power User**
```
1. Rapidly refresh dashboard 10+ times
2. Change configurations quickly
3. Open multiple browser tabs
4. Test during peak market hours
5. Leave running overnight
```

### **Scenario C: Error Conditions**
```
1. Stop container mid-operation
2. Restart container
3. Verify data persistence
4. Check error recovery
```

## **📝 QA Report Format**

After testing, provide:

```markdown
# QA Test Report - Algotrading Agent

## Executive Summary
- Total Tests: X
- Passed: X
- Failed: X  
- Critical Issues: X

## Detailed Results
[List each test case with results]

## Issues Found
### Critical (P1)
- Issue 1: Description
- Issue 2: Description

### High Priority (P2)
- Issue 1: Description

### Medium Priority (P3)
- Issue 1: Description

## Recommendations
1. Fix P1 issues immediately
2. Address P2 issues before production
3. Consider P3 improvements

## Environment Details
- Browser: [Version]
- OS: [System]
- Test Duration: [Time]
- Date: [When tested]
```

## **🚀 Getting Started**

1. **Set up your environment:**
   - Open browser to http://localhost:8080/dashboard
   - Open terminal for docker-compose logs
   - Prepare note-taking document

2. **Start with Phase 1 tests** and work through systematically

3. **Document everything** - even minor observations

4. **Report issues immediately** if you find critical problems

**Ready to start testing?** Begin with Phase 1 and let me know what you discover! 🕵️‍♂️

## **📋 Test Results Log Template**

Copy this template for your testing session:

```markdown
# QA Test Session - [Date]

## Environment Setup
- Dashboard URL: http://localhost:8080/dashboard
- Browser: [Your Browser + Version]
- OS: [Your Operating System]
- Start Time: [Time]

## Phase 1: System Availability
- [ ] Dashboard loads: ✅/❌
- [ ] All UI sections visible: ✅/❌
- [ ] Health endpoint responds: ✅/❌
- [ ] No console errors: ✅/❌
- [ ] Auto-refresh works: ✅/❌
- Notes: ___________________________

## Phase 2: Data Consistency  
- [ ] Total Value = $100,000: ✅/❌
- [ ] Available Cash = $100,000: ✅/❌
- [ ] Daily P&L = $0.00: ✅/❌
- [ ] Position Count = 0: ✅/❌
- [ ] Risk Usage = 0.0%: ✅/❌
- Notes: ___________________________

## Phase 3: Real-Time Data
- [ ] Live logs showing: ✅/❌
- [ ] News items loading: ✅/❌
- [ ] News have titles/sources: ✅/❌
- [ ] Trading decisions section: ✅/❌
- [ ] Timestamps updating: ✅/❌
- Notes: ___________________________

## Phase 4: Configuration
- [ ] Min Confidence update: ✅/❌
- [ ] Success message shown: ✅/❌
- [ ] Stop Loss update: ✅/❌
- [ ] Take Profit update: ✅/❌
- [ ] Settings persist: ✅/❌
- Notes: ___________________________

## Phase 5: Error Handling
- [ ] Invalid config values: ✅/❌
- [ ] 404 page handling: ✅/❌
- [ ] Network disruption: ✅/❌
- [ ] Extended operation: ✅/❌
- Notes: ___________________________

## Phase 6: Browser Compatibility
- [ ] Chrome: ✅/❌
- [ ] Firefox: ✅/❌
- [ ] Mobile view: ✅/❌
- Notes: ___________________________

## Issues Found
1. [Issue description]
2. [Issue description]
3. [Issue description]

## Overall Assessment
- System Stability: ⭐⭐⭐⭐⭐ (1-5 stars)
- User Experience: ⭐⭐⭐⭐⭐ (1-5 stars)
- Performance: ⭐⭐⭐⭐⭐ (1-5 stars)
- Reliability: ⭐⭐⭐⭐⭐ (1-5 stars)

## Recommendations
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]

End Time: [Time]
Total Duration: [Duration]
```