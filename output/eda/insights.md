# EDA Insights — Lecture 6

Source: `cleaned_train.csv` (1,031 prompts).

1. **Balanced classes:** 516 benign and 515 jailbreak prompts.
2. **Length separates classes:** median word count is 36 for benign vs 259 for jailbreak.
3. **Keyword coverage:** at least one selected keyword occurs in 11.0% of benign and 57.9% of jailbreak prompts.
4. **Lexical overlap:** 3 of the top 20 tokens are shared between classes, so many class-specific lexical signals remain.
5. **Scatter relationship:** word count and character count rise together, so they should not be interpreted as independent evidence.
6. **Distribution shape:** histograms and KDEs show strong right skew and long tails, supporting median/IQR alongside mean/SD.
7. **Outliers:** boxplots from the data-preparation audit identify extreme prompt lengths; they are retained because unusually long prompts can be genuine examples.
8. **Visualization decision:** no line plot is created because row order is not a time variable and an arbitrary line would imply a false trend.
