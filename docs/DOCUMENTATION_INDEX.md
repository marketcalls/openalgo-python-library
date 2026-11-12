# OpenAlgo Technical Indicators - Complete Documentation Index

This is a comprehensive index of all documentation files for the OpenAlgo Technical Indicators library.

## Main Documentation Files

### 0. **[INDICATOR_CONVENTIONS.md](INDICATOR_CONVENTIONS.md)** - Seeds, smoothing, windows and NaN
**Purpose**: The seed, smoothing and window convention of each indicator that has
more than one published form, how the indicators treat NaN input, the formulas of the
single-formula indicators, how the conventions are checked in CI, and which functions
cost more than one pass over the bars.

---

### 1. **[README.md](README.md)** - Start Here
**Purpose**: Overview and navigation guide  
**Contents**:
- Documentation structure explanation
- Quick getting started guide
- Key features summary
- Navigation to specific documentation

---

### 2. **[TECHNICAL_INDICATORS_GUIDE.md](TECHNICAL_INDICATORS_GUIDE.md)** - Complete Reference
**Purpose**: Comprehensive guide for all 100+ indicators  
**Contents**:
- Detailed explanations for every indicator
- Mathematical formulas and concepts
- Complete parameter descriptions
- Input/output specifications
- Extensive code examples with explanations
- Trading signal interpretations
- Best practices and optimization tips

**Best for**: Learning about specific indicators, understanding concepts, implementation guidance

---

### 3. **[INDICATORS_QUICK_REFERENCE.md](INDICATORS_QUICK_REFERENCE.md)** - Quick Lookup
**Purpose**: Fast reference for function signatures and parameters  
**Contents**:
- Categorized indicator tables with mandatory/optional parameters marked
- Function signatures with default values
- Quick examples showing default vs custom usage
- Common trading patterns and combinations
- Position sizing examples
- Complete strategy template

**Best for**: Quick syntax lookup, function signatures, parameter checking

---

### 4. **[PARAMETERS_REFERENCE.md](PARAMETERS_REFERENCE.md)** - Parameter Specifications
**Purpose**: Complete parameter documentation with mandatory vs optional clearly marked  
**Contents**:
- 🔴 **MANDATORY** vs 🟢 **OPTIONAL** parameters for every indicator
- Default values for all optional parameters
- Parameter types and descriptions
- Usage examples showing defaults vs custom values
- Tips for parameter selection by use case
- Time horizon recommendations

**Best for**: Understanding which parameters you must provide vs which have defaults, parameter tuning

---

### 5. **[TRADING_STRATEGIES_EXAMPLES.md](TRADING_STRATEGIES_EXAMPLES.md)** - Real-World Examples
**Purpose**: Complete trading strategy implementations  
**Contents**:
- 7 categories of trading strategies:
  - Trend Following Strategies
  - Mean Reversion Strategies  
  - Momentum Strategies
  - Volatility Strategies
  - Volume-Based Strategies
  - Multi-Timeframe Strategies
  - Machine Learning Enhanced Strategies
- Complete code implementations
- Risk management frameworks
- Performance optimization techniques

**Best for**: Learning how to combine indicators, complete strategy examples, risk management

---

## Utility Files

### 6. **[parameters_demo.py](parameters_demo.py)** - Interactive Demo
**Purpose**: Executable demonstration of mandatory vs optional parameters  
**Contents**:
- Live examples showing parameter usage
- Demonstrations of what happens with missing mandatory parameters
- Examples of using defaults vs custom values
- Common mistakes and how to avoid them

**How to use**: `python3 docs/parameters_demo.py`

---

## Documentation Quick Navigation

### I want to...

#### **Learn about a specific indicator**
→ **[TECHNICAL_INDICATORS_GUIDE.md](TECHNICAL_INDICATORS_GUIDE.md)**
- Search for the indicator name
- Find detailed explanation, formula, and examples

#### **Quickly find function syntax**
→ **[INDICATORS_QUICK_REFERENCE.md](INDICATORS_QUICK_REFERENCE.md)**
- Use category tables for fast lookup
- Check parameter requirements

#### **Understand parameter requirements**
→ **[PARAMETERS_REFERENCE.md](PARAMETERS_REFERENCE.md)**
- See which parameters are mandatory vs optional
- Find default values and usage examples

#### **See complete trading strategies**
→ **[TRADING_STRATEGIES_EXAMPLES.md](TRADING_STRATEGIES_EXAMPLES.md)**
- Browse strategy categories
- Copy complete implementations

#### **Get started quickly**
→ **[README.md](README.md)**
- Basic installation and usage
- Overview of library capabilities

#### **Test parameter usage interactively**
→ **[parameters_demo.py](parameters_demo.py)**
- Run the demo script to see examples

---

## Usage Patterns by Experience Level

### **Beginner**
1. Start with **[README.md](README.md)** for overview
2. Use **[PARAMETERS_REFERENCE.md](PARAMETERS_REFERENCE.md)** to understand parameters
3. Run **[parameters_demo.py](parameters_demo.py)** for hands-on examples
4. Refer to **[INDICATORS_QUICK_REFERENCE.md](INDICATORS_QUICK_REFERENCE.md)** for syntax

### **Intermediate** 
1. Use **[INDICATORS_QUICK_REFERENCE.md](INDICATORS_QUICK_REFERENCE.md)** for daily reference
2. Consult **[TECHNICAL_INDICATORS_GUIDE.md](TECHNICAL_INDICATORS_GUIDE.md)** for detailed understanding
3. Explore **[TRADING_STRATEGIES_EXAMPLES.md](TRADING_STRATEGIES_EXAMPLES.md)** for strategy ideas

### **Advanced**
1. **[TRADING_STRATEGIES_EXAMPLES.md](TRADING_STRATEGIES_EXAMPLES.md)** for complex implementations
2. **[TECHNICAL_INDICATORS_GUIDE.md](TECHNICAL_INDICATORS_GUIDE.md)** for optimization tips
3. **[PARAMETERS_REFERENCE.md](PARAMETERS_REFERENCE.md)** for fine-tuning

---

## Finding Specific Information

### **By Indicator Category**

| Category | Best Documentation |
|----------|-------------------|
| **Trend Indicators** | [Technical Guide](TECHNICAL_INDICATORS_GUIDE.md#trend-indicators) → [Quick Reference](INDICATORS_QUICK_REFERENCE.md#-trend-indicators-14) |
| **Momentum Indicators** | [Technical Guide](TECHNICAL_INDICATORS_GUIDE.md#momentum-indicators) → [Quick Reference](INDICATORS_QUICK_REFERENCE.md#-momentum-indicators-5) |
| **Volatility Indicators** | [Technical Guide](TECHNICAL_INDICATORS_GUIDE.md#volatility-indicators) → [Quick Reference](INDICATORS_QUICK_REFERENCE.md#-volatility-indicators-11) |
| **Volume Indicators** | [Technical Guide](TECHNICAL_INDICATORS_GUIDE.md#volume-indicators) → [Quick Reference](INDICATORS_QUICK_REFERENCE.md#-volume-indicators-11) |
| **Oscillators** | [Technical Guide](TECHNICAL_INDICATORS_GUIDE.md#oscillators) → [Quick Reference](INDICATORS_QUICK_REFERENCE.md#-oscillators-10) |
| **Statistical Indicators** | [Technical Guide](TECHNICAL_INDICATORS_GUIDE.md#statistical-indicators) → [Quick Reference](INDICATORS_QUICK_REFERENCE.md#-statistical-indicators-8) |
| **Hybrid Indicators** | [Technical Guide](TECHNICAL_INDICATORS_GUIDE.md#hybrid-indicators) → [Quick Reference](INDICATORS_QUICK_REFERENCE.md#-hybrid-indicators-7) |

### **By Task**

| Task | Best Documentation |
|------|-------------------|
| **Understanding concepts** | [Technical Indicators Guide](TECHNICAL_INDICATORS_GUIDE.md) |
| **Function syntax** | [Quick Reference](INDICATORS_QUICK_REFERENCE.md) |
| **Parameter details** | [Parameters Reference](PARAMETERS_REFERENCE.md) |
| **Trading strategies** | [Trading Strategies Examples](TRADING_STRATEGIES_EXAMPLES.md) |
| **Getting started** | [README](README.md) + [Demo Script](parameters_demo.py) |

### **By Programming Need**

| Need | Best Documentation |
|------|-------------------|
| **Basic usage** | [README.md](README.md) + [parameters_demo.py](parameters_demo.py) |
| **API reference** | [Quick Reference](INDICATORS_QUICK_REFERENCE.md) + [Parameters Reference](PARAMETERS_REFERENCE.md) |
| **Implementation examples** | [Technical Guide](TECHNICAL_INDICATORS_GUIDE.md) + [Trading Strategies](TRADING_STRATEGIES_EXAMPLES.md) |
| **Best practices** | [Technical Guide](TECHNICAL_INDICATORS_GUIDE.md) + [Trading Strategies](TRADING_STRATEGIES_EXAMPLES.md) |

---

## Reading Order Recommendations

### **New to Technical Analysis**
1. [README.md](README.md) - Get oriented
2. [parameters_demo.py](parameters_demo.py) - See it working
3. [TECHNICAL_INDICATORS_GUIDE.md](TECHNICAL_INDICATORS_GUIDE.md) - Learn the concepts
4. [TRADING_STRATEGIES_EXAMPLES.md](TRADING_STRATEGIES_EXAMPLES.md) - Apply the knowledge

### **Experienced with TA, New to OpenAlgo**
1. [README.md](README.md) - Quick overview
2. [INDICATORS_QUICK_REFERENCE.md](INDICATORS_QUICK_REFERENCE.md) - Learn the syntax
3. [PARAMETERS_REFERENCE.md](PARAMETERS_REFERENCE.md) - Understand the parameters
4. [TRADING_STRATEGIES_EXAMPLES.md](TRADING_STRATEGIES_EXAMPLES.md) - See advanced usage

### **Quick Implementation**
1. [INDICATORS_QUICK_REFERENCE.md](INDICATORS_QUICK_REFERENCE.md) - Find the function
2. [PARAMETERS_REFERENCE.md](PARAMETERS_REFERENCE.md) - Check parameters
3. [parameters_demo.py](parameters_demo.py) - See working examples

---

## Key Features Covered

### **Complete Coverage**
- **100+ indicators** documented with examples
- **All parameters** clearly specified (mandatory vs optional)
- **Real-world strategies** with complete implementations
- **Performance tips** and optimization guidance

### **User-Friendly Organization**
- **Multiple access patterns** for different needs
- **Clear navigation** between documents
- **Practical examples** throughout
- **Progressive complexity** from basic to advanced

### **Production Ready**
- **Professional implementations** matching industry standards
- **Error handling** and edge cases covered
- **Performance benchmarks** included
- **Risk management** frameworks provided

---

## Getting Support

### **Documentation Issues**
- Check this index to find the right document
- Use Ctrl+F to search within documents
- Follow the recommended reading order

### **Code Issues**
- Refer to [TECHNICAL_INDICATORS_GUIDE.md](TECHNICAL_INDICATORS_GUIDE.md) for detailed examples
- Run [parameters_demo.py](parameters_demo.py) to see working code
- Check [PARAMETERS_REFERENCE.md](PARAMETERS_REFERENCE.md) for parameter requirements

### **Strategy Questions**
- See [TRADING_STRATEGIES_EXAMPLES.md](TRADING_STRATEGIES_EXAMPLES.md) for complete implementations
- Check [TECHNICAL_INDICATORS_GUIDE.md](TECHNICAL_INDICATORS_GUIDE.md) for best practices

---

**Happy Trading!**