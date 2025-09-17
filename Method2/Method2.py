# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 17:41:25 2025

@author: rijan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import matplotlib.gridspec as gridspec

class EnergyRetrofitOptimizer:
    def __init__(self, discount_rate=0.05, carbon_price=50, budget=100000):
        """
        Initialize the optimizer with financial and environmental parameters
        
        Args:
            discount_rate (float): Discount rate for NPV calculations
            carbon_price (float): Price per ton of CO2 saved ($/ton)
            budget (float): Total available budget ($)
        """
        self.discount_rate = discount_rate
        self.carbon_price = carbon_price
        self.budget = budget
        
        # Set style for plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Example projects with dummy data (users can modify these)
        self.projects = {
            'PV_Installation': {
                'cost': 40000,
                'savings_range': (5000, 7000),  # (min, max) annual savings
                'maintenance': 300,  # annual maintenance cost
                'lifetime': 25,
                'carbon_savings': 8.5,  # tons CO2/year
                'certainty_factor': 0.9  # How certain are the predictions (0-1)
            },
            'Battery_System': {
                'cost': 25000,
                'savings_range': (2000, 3500),
                'maintenance': 400,
                'lifetime': 15,
                'carbon_savings': 2.5,
                'certainty_factor': 0.8
            },
            'HVAC_Modernization': {
                'cost': 35000,
                'savings_range': (4000, 6000),
                'maintenance': 500,
                'lifetime': 20,
                'carbon_savings': 6.2,
                'certainty_factor': 0.85
            },
            'Lighting_Upgrade': {
                'cost': 15000,
                'savings_range': (2500, 4000),
                'maintenance': 100,
                'lifetime': 10,
                'carbon_savings': 3.8,
                'certainty_factor': 0.95
            },
            'Insulation_Upgrade': {
                'cost': 20000,
                'savings_range': (1800, 3000),
                'maintenance': 50,
                'lifetime': 30,
                'carbon_savings': 4.5,
                'certainty_factor': 0.9
            },
            'Water_System_Retrofit': {
                'cost': 12000,
                'savings_range': (1500, 2500),
                'maintenance': 200,
                'lifetime': 15,
                'carbon_savings': 2.2,
                'certainty_factor': 0.85
            },
            'Efficient_Motor_Upgrades': {
                'cost': 18000,
                'savings_range': (2200, 3800),
                'maintenance': 300,
                'lifetime': 12,
                'carbon_savings': 3.5,
                'certainty_factor': 0.88
            }
        }
        
    def calculate_npv(self, project, savings_scenario='expected'):
        """
        Calculate Net Present Value for a project
        
        Args:
            project (dict): Project parameters
            savings_scenario (str): 'min', 'max', or 'expected'
            
        Returns:
            float: Net Present Value
        """
        cost = project['cost']
        maintenance = project['maintenance']
        lifetime = project['lifetime']
        certainty = project['certainty_factor']
        
        # Determine which savings value to use
        min_savings, max_savings = project['savings_range']
        if savings_scenario == 'min':
            annual_savings = min_savings
        elif savings_scenario == 'max':
            annual_savings = max_savings
        else:  # expected
            annual_savings = (min_savings + max_savings) / 2
            
        # Apply certainty factor to savings
        annual_savings *= certainty
        
        # Calculate net annual cash flow (savings - maintenance)
        net_annual_cash_flow = annual_savings - maintenance
        
        # Calculate NPV
        pv_factor = (1 - (1 + self.discount_rate) ** -lifetime) / self.discount_rate
        npv = net_annual_cash_flow * pv_factor - cost
        
        return npv
    
    def calculate_carbon_savings(self, project):
        """
        Calculate total carbon savings over project lifetime
        
        Args:
            project (dict): Project parameters
            
        Returns:
            float: Total carbon savings (tons CO2)
        """
        annual_carbon_savings = project['carbon_savings'] * project['certainty_factor']
        lifetime_carbon_savings = annual_carbon_savings * project['lifetime']
        return lifetime_carbon_savings
    
    def calculate_combined_score(self, project, financial_weight=0.7, carbon_weight=0.3):
        """
        Calculate a combined score considering both financial and carbon criteria
        
        Args:
            project (dict): Project parameters
            financial_weight (float): Weight for financial criteria (0-1)
            carbon_weight (float): Weight for carbon criteria (0-1)
            
        Returns:
            float: Combined score
        """
        # Calculate financial metrics
        npv = self.calculate_npv(project)
        irr = self.calculate_irr(project) if npv > 0 else 0
        
        # Calculate carbon metrics
        carbon_savings = self.calculate_carbon_savings(project)
        carbon_value = carbon_savings * self.carbon_price
        
        # Normalize metrics (simple approach - can be enhanced)
        financial_score = (npv + carbon_value) / project['cost'] if project['cost'] > 0 else 0
        carbon_score = carbon_savings / project['cost'] if project['cost'] > 0 else 0
        
        # Combined score
        combined_score = (financial_weight * financial_score + 
                          carbon_weight * carbon_score)
        
        return combined_score
    
    def calculate_irr(self, project, savings_scenario='expected'):
        """
        Calculate Internal Rate of Return for a project (simplified)
        
        Args:
            project (dict): Project parameters
            savings_scenario (str): 'min', 'max', or 'expected'
            
        Returns:
            float: Internal Rate of Return
        """
        # This is a simplified IRR calculation
        cost = project['cost']
        maintenance = project['maintenance']
        lifetime = project['lifetime']
        certainty = project['certainty_factor']
        
        # Determine which savings value to use
        min_savings, max_savings = project['savings_range']
        if savings_scenario == 'min':
            annual_savings = min_savings
        elif savings_scenario == 'max':
            annual_savings = max_savings
        else:  # expected
            annual_savings = (min_savings + max_savings) / 2
            
        # Apply certainty factor to savings
        annual_savings *= certainty
        
        # Calculate net annual cash flow (savings - maintenance)
        net_annual_cash_flow = annual_savings - maintenance
        
        # Simplified IRR approximation
        if cost > 0:
            irr = (net_annual_cash_flow * lifetime - cost) / (cost * lifetime) * 100
        else:
            irr = 0
            
        return max(0, irr)  # IRR shouldn't be negative
    
    def find_optimal_portfolio(self, financial_weight=0.7, carbon_weight=0.3):
        """
        Find the optimal combination of projects within budget
        
        Args:
            financial_weight (float): Weight for financial criteria (0-1)
            carbon_weight (float): Weight for carbon criteria (0-1)
            
        Returns:
            dict: Optimal portfolio with details
        """
        project_names = list(self.projects.keys())
        best_score = -float('inf')
        best_combination = []
        
        # Generate all possible combinations of projects
        for r in range(1, len(project_names) + 1):
            for combo in combinations(project_names, r):
                total_cost = sum(self.projects[p]['cost'] for p in combo)
                
                # Skip if over budget
                if total_cost > self.budget:
                    continue
                
                # Calculate combined score for this combination
                total_score = 0
                for p in combo:
                    score = self.calculate_combined_score(
                        self.projects[p], financial_weight, carbon_weight
                    )
                    total_score += score
                
                # Update best combination if this one is better
                if total_score > best_score:
                    best_score = total_score
                    best_combination = combo
        
        # Calculate details for the best combination
        total_cost = sum(self.projects[p]['cost'] for p in best_combination)
        total_npv = sum(self.calculate_npv(self.projects[p]) for p in best_combination)
        total_carbon = sum(self.calculate_carbon_savings(self.projects[p]) for p in best_combination)
        
        return {
            'projects': best_combination,
            'total_cost': total_cost,
            'remaining_budget': self.budget - total_cost,
            'total_npv': total_npv,
            'total_carbon_savings': total_carbon,
            'combined_score': best_score
        }
    
    def generate_report(self, portfolio):
        """
        Generate a detailed report of the selected portfolio
        
        Args:
            portfolio (dict): Portfolio details from find_optimal_portfolio
            
        Returns:
            str: Formatted report
        """
        report = "ENERGY RETROFIT PROJECT SELECTION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Budget: ${self.budget:,.2f}\n"
        report += f"Total Cost of Selected Projects: ${portfolio['total_cost']:,.2f}\n"
        report += f"Remaining Budget: ${portfolio['remaining_budget']:,.2f}\n\n"
        
        report += "SELECTED PROJECTS:\n"
        for i, project in enumerate(portfolio['projects'], 1):
            p_data = self.projects[project]
            npv = self.calculate_npv(p_data)
            irr = self.calculate_irr(p_data)
            carbon = self.calculate_carbon_savings(p_data)
            
            report += f"{i}. {project.replace('_', ' ').title()}\n"
            report += f"   Cost: ${p_data['cost']:,.2f}\n"
            report += f"   NPV: ${npv:,.2f}\n"
            report += f"   IRR: {irr:.2f}%\n"
            report += f"   Carbon Savings: {carbon:.2f} tons CO2\n"
            report += f"   Lifetime: {p_data['lifetime']} years\n"
            report += f"   Certainty Factor: {p_data['certainty_factor']*100:.1f}%\n\n"
        
        report += "PORTFOLIO SUMMARY:\n"
        report += f"Total NPV: ${portfolio['total_npv']:,.2f}\n"
        report += f"Total Carbon Savings: {portfolio['total_carbon_savings']:,.2f} tons CO2\n"
        report += f"Combined Score: {portfolio['combined_score']:.4f}\n"
        
        return report
    
    def create_visualizations(self, portfolio):
        """
        Create comprehensive visualizations for project analysis
        
        Args:
            portfolio (dict): Portfolio details from find_optimal_portfolio
        """
        # Prepare data for visualizations
        project_names = []
        costs = []
        npvs = []
        irrs = []
        carbon_savings = []
        combined_scores = []
        lifetimes = []
        certainty_factors = []
        
        for name, data in self.projects.items():
            project_names.append(name.replace('_', '\n'))
            costs.append(data['cost'])
            npvs.append(self.calculate_npv(data))
            irrs.append(self.calculate_irr(data))
            carbon_savings.append(self.calculate_carbon_savings(data))
            combined_scores.append(self.calculate_combined_score(data))
            lifetimes.append(data['lifetime'])
            certainty_factors.append(data['certainty_factor'] * 100)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Energy Retrofit Project Analysis', fontsize=16, fontweight='bold')
        
        # Create a grid for subplots
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # 1. Cost vs NPV bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(project_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, costs, width, label='Cost', color='lightcoral')
        ax1.set_ylabel('Cost ($)', fontweight='bold')
        ax1.set_title('Project Cost vs NPV')
        ax1.set_xticks(x)
        ax1.set_xticklabels(project_names, rotation=45, ha='right')
        
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, npvs, width, label='NPV', color='lightgreen')
        ax2.set_ylabel('NPV ($)', fontweight='bold')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height/1000:.0f}k', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height/1000:.0f}k', ha='center', va='bottom', fontsize=8)
        
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # 2. IRR vs Carbon Savings scatter plot
        ax3 = fig.add_subplot(gs[0, 1])
        scatter = ax3.scatter(irrs, carbon_savings, s=np.array(costs)/100, 
                             c=combined_scores, cmap='viridis', alpha=0.7)
        
        ax3.set_xlabel('IRR (%)', fontweight='bold')
        ax3.set_ylabel('Carbon Savings (tons CO2)', fontweight='bold')
        ax3.set_title('IRR vs Carbon Savings (Size = Cost, Color = Combined Score)')
        
        # Add project labels to points
        for i, name in enumerate(project_names):
            ax3.annotate(name, (irrs[i], carbon_savings[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Combined Score', fontweight='bold')
        
        # 3. Combined Score bar chart
        ax4 = fig.add_subplot(gs[0, 2])
        colors = ['lightblue' if name.replace('\n', '_') not in portfolio['projects'] 
                 else 'gold' for name in project_names]
        
        bars = ax4.bar(project_names, combined_scores, color=colors)
        ax4.set_ylabel('Combined Score', fontweight='bold')
        ax4.set_title('Project Combined Scores (Selected in Gold)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Portfolio cost breakdown pie chart
        ax5 = fig.add_subplot(gs[1, 0])
        portfolio_costs = [self.projects[p]['cost'] for p in portfolio['projects']]
        portfolio_labels = [p.replace('_', '\n') for p in portfolio['projects']]
        
        wedges, texts, autotexts = ax5.pie(portfolio_costs, labels=portfolio_labels, autopct='%1.1f%%',
                                           startangle=90, colors=sns.color_palette("Set3", len(portfolio_costs)))
        ax5.set_title('Portfolio Cost Distribution')
        
        # 5. Portfolio NPV vs Carbon Savings
        ax6 = fig.add_subplot(gs[1, 1])
        portfolio_npvs = [self.calculate_npv(self.projects[p]) for p in portfolio['projects']]
        portfolio_carbons = [self.calculate_carbon_savings(self.projects[p]) for p in portfolio['projects']]
        
        bars = ax6.bar(portfolio_labels, portfolio_npvs, color='lightgreen', label='NPV')
        ax6.set_ylabel('NPV ($)', fontweight='bold')
        ax6.set_title('Portfolio: NPV by Project')
        ax6.tick_params(axis='x', rotation=45)
        
        ax7 = ax6.twinx()
        ax7.plot(portfolio_labels, portfolio_carbons, color='darkblue', 
                marker='o', linewidth=2, markersize=8, label='Carbon Savings')
        ax7.set_ylabel('Carbon Savings (tons CO2)', fontweight='bold', color='darkblue')
        ax7.tick_params(axis='y', labelcolor='darkblue')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height/1000:.0f}k', ha='center', va='bottom', fontsize=8)
        
        # 6. Certainty vs Lifetime scatter plot
        ax8 = fig.add_subplot(gs[1, 2])
        scatter = ax8.scatter(lifetimes, certainty_factors, s=np.array(costs)/100, 
                             c=combined_scores, cmap='plasma', alpha=0.7)
        
        ax8.set_xlabel('Lifetime (years)', fontweight='bold')
        ax8.set_ylabel('Certainty Factor (%)', fontweight='bold')
        ax8.set_title('Lifetime vs Certainty (Size = Cost, Color = Combined Score)')
        
        # Add project labels to points
        for i, name in enumerate(project_names):
            ax8.annotate(name, (lifetimes[i], certainty_factors[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax8)
        cbar.set_label('Combined Score', fontweight='bold')
        
        # 7. Budget utilization
        ax9 = fig.add_subplot(gs[2, 0])
        used_budget = portfolio['total_cost']
        remaining_budget = self.budget - used_budget
        
        wedges, texts, autotexts = ax9.pie([used_budget, remaining_budget], 
                                           labels=['Used Budget', 'Remaining Budget'], 
                                           autopct='%1.1f%%', startangle=90,
                                           colors=['lightcoral', 'lightgreen'])
        ax9.set_title('Budget Utilization')
        
        # 8. Portfolio summary metrics
        ax10 = fig.add_subplot(gs[2, 1:])
        ax10.axis('off')
        
        summary_text = (
            f"Portfolio Summary:\n\n"
            f"Total Projects: {len(portfolio['projects'])}\n"
            f"Total Cost: ${portfolio['total_cost']:,.0f}\n"
            f"Remaining Budget: ${portfolio['remaining_budget']:,.0f}\n"
            f"Total NPV: ${portfolio['total_npv']:,.0f}\n"
            f"Total Carbon Savings: {portfolio['total_carbon_savings']:,.0f} tons CO₂\n"
            f"Combined Score: {portfolio['combined_score']:.4f}\n\n"
            f"Selected Projects:\n" + "\n".join([f"- {p.replace('_', ' ')}" for p in portfolio['projects']])
        )
        
        ax10.text(0.1, 0.9, summary_text, transform=ax10.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig('energy_retrofit_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create a separate figure for savings uncertainty
        fig2, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Savings range visualization
        min_savings = [self.projects[p]['savings_range'][0] for p in self.projects]
        max_savings = [self.projects[p]['savings_range'][1] for p in self.projects]
        expected_savings = [(min_s + max_s) / 2 for min_s, max_s in zip(min_savings, max_savings)]
        
        x = np.arange(len(project_names))
        
        axes[0].bar(x, expected_savings, yerr=[np.array(expected_savings) - np.array(min_savings), 
                                              np.array(max_savings) - np.array(expected_savings)],
                   capsize=5, color='lightblue', alpha=0.7)
        axes[0].set_ylabel('Annual Savings ($)', fontweight='bold')
        axes[0].set_title('Annual Savings with Uncertainty Range')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(project_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(expected_savings):
            axes[0].text(i, v + 100, f'${v:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Maintenance cost vs savings
        maintenance_costs = [self.projects[p]['maintenance'] for p in self.projects]
        
        axes[1].bar(x, expected_savings, label='Expected Savings', color='lightgreen')
        axes[1].bar(x, maintenance_costs, label='Maintenance Cost', color='lightcoral')
        axes[1].set_ylabel('Amount ($)', fontweight='bold')
        axes[1].set_title('Expected Savings vs Maintenance Cost')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(project_names, rotation=45, ha='right')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('savings_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize the optimizer with a $100,000 budget
    optimizer = EnergyRetrofitOptimizer(budget=100000)
    
    # Find the optimal portfolio (70% weight to financial, 30% to carbon)
    portfolio = optimizer.find_optimal_portfolio(financial_weight=0.7, carbon_weight=0.3)
    
    # Generate and print the report
    report = optimizer.generate_report(portfolio)
    print(report)
    
    # Create visualizations
    optimizer.create_visualizations(portfolio)
    
    # Print individual project analysis for comparison
    print("\nINDIVIDUAL PROJECT ANALYSIS:")
    print("Project".ljust(25) + "Cost".ljust(10) + "NPV".ljust(12) + "IRR".ljust(8) + "Carbon".ljust(10) + "Score")
    print("-" * 70)
    
    for name, data in optimizer.projects.items():
        npv = optimizer.calculate_npv(data)
        irr = optimizer.calculate_irr(data)
        carbon = optimizer.calculate_carbon_savings(data)
        score = optimizer.calculate_combined_score(data)
        
        print(f"{name.replace('_', ' ').ljust(25)} "
              f"${data['cost']:,.0f}".ljust(10) +
              f"${npv:,.0f}".ljust(12) +
              f"{irr:.1f}%".ljust(8) +
              f"{carbon:.1f}".ljust(10) +
              f"{score:.4f}")