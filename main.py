from src.credit_risk_analyser import CreditRiskAnalyser

def main():
    file_path = 'data/credit_risk.csv'
    new_file_path = 'cleaned_credit_risk.csv'
    credit_risk_analyser = CreditRiskAnalyser(file_path)
    credit_risk_analyser.analyse_credit_risk(new_file_path)
    
if __name__ == "__main__":
    main()