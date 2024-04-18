from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import io
import base64
import time

app = Flask(__name__)

# Fonction pour calculer D_AB
def calculate_D_AB(Xa, a_AB, a_BA, λ_a, λ_b, q_a, q_b, D_AB0, D_BA0, T):
    Xb = 1 - Xa  # Fraction molaire de B
    D = Xa*(D_BA0) + Xb*np.log(D_AB0) + \
        2*(Xa*np.log(Xa+(Xb*λ_b)/λ_a)+Xb*np.log(Xb+(Xa*λ_a)/λ_b)) + \
        2*Xa*Xb*((λ_a/(Xa*λ_a+Xb*λ_b))*(1-(λ_a/λ_b)) +
                 (λ_b/(Xa*λ_a+Xb*λ_b))*(1-(λ_b/λ_a))) + \
        Xb*q_a*((1-((Xb*q_b*np.exp(-a_BA/T))/(Xa*q_a+Xb*q_b*np.exp(-a_BA/T)))**2)*(-a_BA/T)+(1-((Xb*q_b)/(Xb*q_b+Xa*q_a*np.exp(-a_AB/T)))**2)*np.exp(-a_AB/T)*(-a_AB/T)) + \
        Xa*q_b*((1-((Xa*q_a*np.exp(-a_AB/T))/(Xa*q_a*np.exp(-a_AB/T)+Xb*q_b))**2)*(-a_AB/T)+(1-((Xa*q_a)/(Xa*q_a+Xb*q_b*np.exp(-a_BA/T)))**2)*np.exp(-a_BA/T)*(-a_BA/T))
    # Calcul de D_AB
    return np.exp(D)

# Fonction objectif pour la minimisation
def objective(params, Xa_values, D_AB_exp, λ_a, λ_b, q_a, q_b, D_AB0, D_BA0, T):
    a_AB, a_BA = params
    D_AB_calculated = calculate_D_AB(Xa_values, a_AB, a_BA, λ_a, λ_b, q_a, q_b, D_AB0, D_BA0, T)
    return np.sum((D_AB_calculated - D_AB_exp)**2)

@app.route('/')
def input_data():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    if request.method == 'POST':
        D_AB_exp = float(request.form['D_AB_exp'])
        T = float(request.form['T'])
        Xa = float(request.form['Xa'])
        λ_a = eval(request.form['λ_a'])
        λ_b = eval(request.form['λ_b'])
        q_a = float(request.form['q_a'])
        q_b = float(request.form['q_b'])
        D_AB0 = float(request.form['D_AB0'])
        D_BA0 = float(request.form['D_BA0'])
        
        # Paramètres initiaux
        params_initial = [0, 0]

        # Tolerance
        tolerance = 1e-12

        # Nombre maximal d'itérations
        max_iterations = 1000
        iteration = 0

        # Temps de départ
        start_time = time.time()

        # Boucle d'ajustement des paramètres
        while iteration < max_iterations:
            # Minimisation de l'erreur
            result = minimize(objective, params_initial, args=(Xa, D_AB_exp, λ_a, λ_b, q_a, q_b, D_AB0, D_BA0, T), method='Nelder-Mead')
            # Paramètres optimisés
            a_AB_opt, a_BA_opt = result.x
            # Calcul de D_AB avec les paramètres optimisés
            D_AB_opt = calculate_D_AB(Xa, a_AB_opt, a_BA_opt, λ_a, λ_b, q_a, q_b, D_AB0, D_BA0, T)
            # Calcul de l'erreur
            error = np.abs(D_AB_opt - D_AB_exp)
            # Vérifier si la différence entre les paramètres optimisés est inférieure à la tolérance
            if np.max(np.abs(np.array(params_initial) - np.array([a_AB_opt, a_BA_opt]))) < tolerance:
                break
            # Mise à jour des paramètres initiaux
            params_initial = [a_AB_opt, a_BA_opt]
            # Incrémentation du nombre d'itérations
            iteration += 1

        # Temps d'exécution
        execution_time = time.time() - start_time

        # Générer la courbe
        Xa_values = np.linspace(0, 0.7, 100)  # Fraction molaire de A
        D_AB_values = calculate_D_AB(Xa_values, a_AB_opt, a_BA_opt, λ_a, λ_b, q_a, q_b, D_AB0, D_BA0, T)
        plt.plot(Xa_values, D_AB_values)
        plt.xlabel('Fraction molaire de A')
        plt.ylabel('Coefficient de diffusion (cm^2/s)')
        plt.title('Variation du coefficient de diffusion en fonction du fraction molaire')
        plt.grid(True)
        
        # Convertir le graphique en une représentation base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        graph = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        # Affichage des résultats
        return render_template('result.html', a_AB_opt=a_AB_opt, a_BA_opt=a_BA_opt, D_AB_opt=D_AB_opt, error=error, iteration=iteration, execution_time=execution_time, graph=graph)

if __name__ == '__main__':
    app.run(debug=True)
