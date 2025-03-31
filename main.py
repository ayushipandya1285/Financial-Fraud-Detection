import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")


# Initialize session state
def initialize_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = []
    if 'new_transactions' not in st.session_state:
        st.session_state.new_transactions = []
    if 'fraud_alerts' not in st.session_state:
        st.session_state.fraud_alerts = []
    if 'admin_metrics' not in st.session_state:
        st.session_state.admin_metrics = {
            'total_transactions': 0,
            'fraud_cases': 0,
            'true_positives': 0,
            'active_alerts': 0
        }
    if 'all_transactions' not in st.session_state:
        st.session_state.all_transactions = []
    if 'account_balances' not in st.session_state:
        st.session_state.account_balances = {}
    if 'transaction_graph' not in st.session_state:
        st.session_state.transaction_graph = nx.Graph()


# Load dataset
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("update_data.csv")

        # Convert timestamp
        df['timestamp'] = '2025-01-01 ' + df['timestamp']
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
        df.sort_values(by='timestamp', inplace=True)

        # Ensure required columns exist
        required_columns = ['customer_id', 'dest_id', 'kyc_verified', 'customer_risk_score',
                            'account_age_days', 'is_pep', 'oldbalanceOrg', 'newbalanceOrig',
                            'oldbalanceDest', 'newbalanceDest', 'type', 'isFraud']

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Create lookup table and initialize balances
        df_for_lookup = df[required_columns + ['amount', 'timestamp']].drop_duplicates()

        # Initialize account balances
        for _, row in df_for_lookup.iterrows():
            customer_id = row['customer_id']
            dest_id = row['dest_id']
            st.session_state.account_balances[customer_id] = row['newbalanceOrig']
            st.session_state.account_balances[dest_id] = row['newbalanceDest']

        # Build transaction graph
        for _, row in df_for_lookup.iterrows():
            st.session_state.transaction_graph.add_edge(row['customer_id'], row['dest_id'],
                                                        weight=row['amount'],
                                                        timestamp=row['timestamp'])

        return df, df_for_lookup
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


# Time Series Anomaly Detection
def detect_time_series_anomalies(transaction, window=5):
    try:
        # Get recent transactions for this customer
        customer_transactions = [t for t in st.session_state.all_transactions
                                 if t['customer_id'] == transaction['customer_id']]

        if len(customer_transactions) >= window:
            # Create a temporary DataFrame for analysis
            temp_df = pd.DataFrame(customer_transactions[-window:] + [transaction])
            temp_df['amount'] = temp_df['amount'].astype(float)

            # Calculate rolling statistics
            temp_df['rolling_mean'] = temp_df['amount'].rolling(window=window).mean()
            temp_df['rolling_std'] = temp_df['amount'].rolling(window=window).std()
            temp_df['z_score'] = (temp_df['amount'] - temp_df['rolling_mean']) / temp_df['rolling_std']

            # Check if current transaction is an anomaly
            is_anomaly = abs(temp_df.iloc[-1]['z_score']) > 3 if not np.isnan(temp_df.iloc[-1]['z_score']) else False
            return is_anomaly
        return False
    except Exception as e:
        st.error(f"Error in time series anomaly detection: {str(e)}")
        return False


# Graph-Based Fraud Detection
def detect_graph_anomalies(transaction):
    try:
        G = st.session_state.transaction_graph

        # Check if this is a new edge or an existing one with unusual amount
        if G.has_edge(transaction['customer_id'], transaction['dest_id']):
            edge_data = G.get_edge_data(transaction['customer_id'], transaction['dest_id'])
            prev_amount = edge_data.get('weight', 0)

            # If amount is significantly different from previous transactions
            if transaction['amount'] > 2 * prev_amount:
                return True
        else:
            # New connection - check if either party has many connections (potential mule)
            customer_degree = G.degree(transaction['customer_id']) if transaction['customer_id'] in G else 0
            dest_degree = G.degree(transaction['dest_id']) if transaction['dest_id'] in G else 0

            if customer_degree > 5 or dest_degree > 5:
                return True
        return False
    except Exception as e:
        st.error(f"Error in graph anomaly detection: {str(e)}")
        return False


# Initialize models
@st.cache_resource
def initialize_models(df):
    try:
        if df.empty:
            raise ValueError("Empty dataset provided for model training")

        # Encode categorical variables
        label_encoders = {}
        categorical_cols = ['type', 'kyc_verified', 'is_pep']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        # Prepare features and target
        features_to_remove = ['oldbalanceOrg', 'amount', 'newbalanceOrig',
                              'oldbalanceDest', 'newbalanceDest', 'customer_id',
                              'dest_id', 'timestamp', 'isFraud']
        X = df.drop(columns=features_to_remove + (['timestamp'] if 'timestamp' in df.columns else []))
        y = df['isFraud']

        # Normalize numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Train models
        supervised_model = RandomForestClassifier(n_estimators=100, random_state=42)
        supervised_model.fit(X_resampled, y_resampled)

        mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                                  solver='adam', max_iter=200, random_state=42)
        mlp_model.fit(X_resampled, y_resampled)

        unsupervised_model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        unsupervised_model.fit(X_resampled)

        return {
            'supervised_model': supervised_model,
            'mlp_model': mlp_model,
            'unsupervised_model': unsupervised_model,
            'label_encoders': label_encoders,
            'scaler': scaler,
            'feature_columns': X.columns
        }
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return None


# Update account balances
def update_balances(customer_id, dest_id, amount, transaction_type):
    try:
        # Get current balances
        customer_balance = st.session_state.account_balances.get(customer_id, 0)
        dest_balance = st.session_state.account_balances.get(dest_id, 0)

        # Update based on transaction type
        if transaction_type in ['CASH-OUT', 'PAYMENT', 'TRANSFER']:
            st.session_state.account_balances[customer_id] = customer_balance - amount
            st.session_state.account_balances[dest_id] = dest_balance + amount
        elif transaction_type == 'CASH-IN':
            st.session_state.account_balances[customer_id] = customer_balance + amount

        # Update transaction graph
        if st.session_state.transaction_graph.has_edge(customer_id, dest_id):
            # Update edge weight (average amount)
            current_weight = st.session_state.transaction_graph[customer_id][dest_id]['weight']
            new_weight = (current_weight + amount) / 2
            st.session_state.transaction_graph[customer_id][dest_id]['weight'] = new_weight
        else:
            # Add new edge
            st.session_state.transaction_graph.add_edge(customer_id, dest_id,
                                                        weight=amount,
                                                        timestamp=datetime.now())
    except Exception as e:
        st.error(f"Error updating balances: {str(e)}")


# Business rules
def apply_business_rules(transaction):
    rules_triggered = []
    try:
        customer_balance = st.session_state.account_balances.get(transaction['customer_id'], 0)

        if transaction['amount'] > 10000 and customer_balance == 0:
            rules_triggered.append("Large transfer from empty account")
        if customer_balance - transaction['amount'] < 0:
            rules_triggered.append("Insufficient funds")
        if transaction['amount'] > 50000:
            rules_triggered.append("High-value transaction")
        if transaction['amount'] > 0.9 * customer_balance:
            rules_triggered.append("Balance drainage (>90% of balance)")
    except KeyError:
        pass
    return rules_triggered


# KYC and AML rules
def apply_kyc_rules(transaction):
    kyc_issues = []
    try:
        if transaction.get('kyc_verified', 0) == 0:
            kyc_issues.append("KYC not verified")
        if transaction.get('is_pep', 0) == 1 and transaction['amount'] > 10000:
            kyc_issues.append("High-value transaction by PEP")
        if transaction.get('customer_risk_score', 0) > 0.8:
            kyc_issues.append("High-risk customer")
        if transaction.get('account_age_days', 0) < 30 and transaction['amount'] > 5000:
            kyc_issues.append("New Account Monitoring")
    except KeyError:
        pass
    return kyc_issues


# Fraud detection
def detect_fraud(transaction, models):
    try:
        if not models:
            raise ValueError("Models not initialized")

        # Prepare input data with current balances
        transaction.update({
            'oldbalanceOrg': st.session_state.account_balances.get(transaction['customer_id'], 0),
            'oldbalanceDest': st.session_state.account_balances.get(transaction['dest_id'], 0)
        })

        # Prepare model input
        input_data = {}
        for col in models['feature_columns']:
            if col in transaction:
                input_data[col] = transaction[col]
            elif col in models['label_encoders']:
                input_data[col] = 0  # Default encoded value
            else:
                input_data[col] = 0  # Default numerical value

        input_df = pd.DataFrame([input_data])

        # Encode categorical variables
        for col in models['label_encoders']:
            if col in input_df.columns:
                input_df[col] = models['label_encoders'][col].transform(input_df[col])

        # Scale features
        input_scaled = models['scaler'].transform(input_df[models['feature_columns']])

        # Get predictions
        supervised_pred = models['supervised_model'].predict(input_scaled)[0]
        mlp_pred = models['mlp_model'].predict(input_scaled)[0]
        unsupervised_score = models['unsupervised_model'].decision_function(input_scaled)[0]
        unsupervised_pred = int(unsupervised_score < 0)

        # Graph and time series analysis
        graph_anomaly = detect_graph_anomalies(transaction)
        time_anomaly = detect_time_series_anomalies(transaction)

        # Calculate hybrid risk score
        hybrid_score = (0.35 * supervised_pred +
                        0.25 * mlp_pred +
                        0.2 * unsupervised_pred +
                        0.1 * graph_anomaly +
                        0.1 * time_anomaly)

        hybrid_pred = int(hybrid_score > 0.6)

        # Apply rules
        business_rules = apply_business_rules(transaction)
        kyc_rules = apply_kyc_rules(transaction)

        # Calculate final risk score
        risk_score = min(0.4 * hybrid_score +
                         0.3 * len(business_rules) / 3 +
                         0.3 * len(kyc_rules) / 4, 1.0)

        is_fraud = hybrid_pred or (risk_score > 0.65) or len(business_rules) > 1

        return {
            'is_fraud': is_fraud,
            'risk_score': risk_score,
            'business_rules': business_rules,
            'kyc_rules': kyc_rules,
            'predictions': {
                'supervised': supervised_pred,
                'mlp': mlp_pred,
                'unsupervised': unsupervised_pred,
                'graph_anomaly': graph_anomaly,
                'time_anomaly': time_anomaly
            },
            'hybrid': hybrid_pred
        }
    except Exception as e:
        st.error(f"Error in fraud detection: {str(e)}")
        return {
            'is_fraud': False,
            'risk_score': 0.0,
            'business_rules': [],
            'kyc_rules': [],
            'predictions': {
                'supervised': 0,
                'mlp': 0,
                'unsupervised': 0,
                'graph_anomaly': False,
                'time_anomaly': False
            },
            'hybrid': 0
        }


# User Panel updates
def display_user_panel(df_for_lookup, models):
    st.title("User Dashboard")

    try:
        if df_for_lookup.empty:
            st.error("No user data available")
            return

        # User selection
        user_options = df_for_lookup['customer_id'].unique()
        if len(user_options) == 0:
            st.error("No users found in the system")
            return

        user_id = st.selectbox("Select your profile", user_options)

        # Get user data
        user_data = df_for_lookup[df_for_lookup['customer_id'] == user_id]
        if user_data.empty:
            st.error("User data not found")
            return

        user_data = user_data.iloc[0]

        # Get current balance
        current_balance = st.session_state.account_balances.get(user_id, user_data.get('newbalanceOrig', 0))

        # User information
        st.subheader("User Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Customer ID", user_id)
            st.metric("Account Age", f"{user_data.get('account_age_days', 'N/A')} days")
        with col2:
            st.metric("KYC Status", "Verified" if user_data.get('kyc_verified', 0) else "Not Verified")
            st.metric("Risk Score", f"{user_data.get('customer_risk_score', 0) * 100:.0f}/100")
        with col3:
            st.metric("PEP Status", "Yes" if user_data.get('is_pep', 0) else "No")
            st.metric("Current Balance", f"${current_balance:,.2f}")

        # Network information
        if user_id in st.session_state.transaction_graph:
            degree = st.session_state.transaction_graph.degree(user_id)
            st.info(f"Your account is connected to {degree} other accounts in our network")

        # Submit new transaction
        st.subheader("Submit New Transaction")
        with st.form("transaction_form"):
            trans_type = st.selectbox("Transaction Type",
                                      ['CASH-IN', 'CASH-OUT', 'PAYMENT', 'TRANSFER'])

            # Handle max_value properly for different transaction types
            max_amount = current_balance if trans_type in ['CASH-OUT', 'PAYMENT', 'TRANSFER'] else float('inf')
            max_amount = min(max_amount, 1e12)  # Set reasonable upper limit

            amount = st.number_input("Amount",
                                     min_value=0.01,
                                     value=100.0,
                                     max_value=max_amount)

            dest_options = df_for_lookup['dest_id'].unique()
            dest_account = st.selectbox("Destination Account", dest_options)

            description = st.text_input("Description")

            submitted = st.form_submit_button("Submit Transaction")

            if submitted:
                try:
                    # Calculate new balances
                    new_balance_orig = current_balance
                    new_balance_dest = st.session_state.account_balances.get(dest_account, 0)

                    if trans_type == 'CASH-IN':
                        new_balance_orig += amount
                    elif trans_type in ['CASH-OUT', 'PAYMENT', 'TRANSFER']:
                        new_balance_orig -= amount
                        new_balance_dest += amount

                    # Create transaction record
                    transaction = {
                        'timestamp': datetime.now(),
                        'customer_id': user_id,
                        'dest_id': dest_account,
                        'type': trans_type,
                        'amount': amount,
                        'description': description,
                        'status': 'Pending Review',
                        'oldbalanceOrg': current_balance,
                        'newbalanceOrig': new_balance_orig,
                        'oldbalanceDest': st.session_state.account_balances.get(dest_account, 0),
                        'newbalanceDest': new_balance_dest,
                        'kyc_verified': user_data.get('kyc_verified', 0),
                        'customer_risk_score': user_data.get('customer_risk_score', 0),
                        'account_age_days': user_data.get('account_age_days', 0),
                        'is_pep': user_data.get('is_pep', 0)
                    }

                    # Convert type to encoded value
                    if models and 'label_encoders' in models and 'type' in models['label_encoders']:
                        transaction['type_encoded'] = models['label_encoders']['type'].transform([trans_type])[0]

                    # Fraud detection
                    fraud_result = detect_fraud(transaction, models) if models else {
                        'is_fraud': False,
                        'risk_score': 0.0,
                        'business_rules': [],
                        'kyc_rules': []
                    }

                    # Update transaction with fraud info
                    transaction.update({
                        'is_fraud': fraud_result['is_fraud'],
                        'risk_score': fraud_result['risk_score'],
                        'business_rules': fraud_result['business_rules'],
                        'kyc_rules': fraud_result['kyc_rules'],
                        'status': 'Flagged' if fraud_result['is_fraud'] else 'Pending Review',
                        'predictions': fraud_result['predictions']
                    })

                    # Add to session state
                    st.session_state.data.append(transaction)
                    st.session_state.new_transactions.append(transaction)
                    st.session_state.all_transactions.append(transaction)
                    st.session_state.admin_metrics['total_transactions'] += 1

                    if fraud_result['is_fraud']:
                        st.session_state.fraud_alerts.append(transaction)
                        st.session_state.admin_metrics['active_alerts'] += 1
                        st.session_state.admin_metrics['fraud_cases'] += 1
                        st.error("⚠ This transaction was flagged as potentially fraudulent!")
                    else:
                        # Update balances immediately if not fraud
                        update_balances(
                            user_id,
                            dest_account,
                            amount,
                            trans_type
                        )
                        st.success("✅ Transaction submitted for review")

                except Exception as e:
                    st.error(f"Error submitting transaction: {str(e)}")

        # View recent transactions
        st.subheader("Your Recent Transactions")
        if st.session_state.all_transactions:
            user_transactions = [t for t in st.session_state.all_transactions
                                 if t.get('customer_id') == user_id]
            if user_transactions:
                # Show last 10 transactions with selected columns
                recent_trans = pd.DataFrame(user_transactions[-10:])

                # Select and format columns
                display_cols = ['timestamp', 'type', 'amount', 'status', 'dest_id']
                display_data = []

                for trans in user_transactions[-10:]:
                    display_data.append({
                        'timestamp': trans['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(trans['timestamp'],
                                                                                                 'strftime') else str(
                            trans['timestamp']),
                        'type': trans['type'],
                        'amount': f"${trans['amount']:,.2f}",
                        'status': trans['status'],
                        'destination': trans['dest_id'] if trans['type'] in ['TRANSFER', 'PAYMENT'] else 'N/A'
                    })

                st.dataframe(pd.DataFrame(display_data))
            else:
                st.info("No transactions found for this user")
        else:
            st.info("No transactions found")

    except Exception as e:
        st.error(f"Error in user panel: {str(e)}")


# Admin Panel updates
def display_admin_panel():
    st.title("Admin Dashboard")

    try:
        # Metrics overview
        st.subheader("System Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", st.session_state.admin_metrics['total_transactions'])
        with col2:
            st.metric("Fraud Cases", st.session_state.admin_metrics['fraud_cases'])
        with col3:
            st.metric("True Positives", st.session_state.admin_metrics['true_positives'])
        with col4:
            st.metric("Active Alerts", st.session_state.admin_metrics['active_alerts'])

        # Network visualization
        st.subheader("Transaction Network Overview")
        if st.session_state.transaction_graph.number_of_nodes() > 0:
            st.write(f"Network contains {st.session_state.transaction_graph.number_of_nodes()} accounts and "
                     f"{st.session_state.transaction_graph.number_of_edges()} transactions")

            # Calculate some network metrics
            degrees = dict(st.session_state.transaction_graph.degree())
            avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
            st.write(f"Average connections per account: {avg_degree:.1f}")

            # Identify potential mule accounts (high degree)
            if degrees:
                max_degree = max(degrees.values())
                mule_candidates = [k for k, v in degrees.items() if v == max_degree]
                st.warning(
                    f"Potential money mule accounts (most connections: {max_degree}): {', '.join(mule_candidates[:3])}{'...' if len(mule_candidates) > 3 else ''}")
        else:
            st.info("No transaction network data available")

        # Transaction history
        st.subheader("Complete Transaction History")
        if st.session_state.all_transactions:
            all_trans_df = pd.DataFrame(st.session_state.all_transactions)

            # Combine business rules and KYC issues into Risk Factors
            all_trans_df['risk_factors'] = all_trans_df.apply(lambda x:
                                                              ', '.join([*x.get('business_rules', []),
                                                                         *x.get('kyc_rules', [])]), axis=1)

            # Format columns
            display_cols = ['timestamp', 'customer_id', 'dest_id', 'type', 'amount',
                            'status', 'is_fraud', 'risk_score', 'risk_factors']
            display_cols = [col for col in display_cols if col in all_trans_df.columns]

            # Format values
            if 'timestamp' in all_trans_df.columns:
                all_trans_df['timestamp'] = all_trans_df['timestamp'].apply(
                    lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if hasattr(x, 'strftime') else str(x))
            if 'risk_score' in all_trans_df.columns:
                all_trans_df['risk_score'] = all_trans_df['risk_score'].apply(lambda x: f"{x * 100:.1f}%")
            if 'is_fraud' in all_trans_df.columns:
                all_trans_df['is_fraud'] = all_trans_df['is_fraud'].apply(lambda x: "Yes" if x else "No")

            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                filter_status = st.selectbox("Filter by Status",
                                             ['All'] + list(all_trans_df['status'].unique()))
            with col2:
                filter_fraud = st.selectbox("Filter by Fraud Status",
                                            ['All', 'Yes', 'No'])

            # Apply filters
            filtered_df = all_trans_df.copy()
            if filter_status != 'All':
                filtered_df = filtered_df[filtered_df['status'] == filter_status]
            if filter_fraud == 'Yes':
                filtered_df = filtered_df[filtered_df['is_fraud'] == "Yes"]
            elif filter_fraud == 'No':
                filtered_df = filtered_df[filtered_df['is_fraud'] == "No"]

            st.dataframe(filtered_df[display_cols])

            # CSV Export
            st.download_button(
                label="Download Transactions as CSV",
                data=filtered_df.to_csv(index=False).encode('utf-8'),
                file_name='transactions.csv',
                mime='text/csv'
            )
        else:
            st.info("No transactions in history")

        # New transactions section
        st.subheader("New Transactions for Review")
        if st.session_state.new_transactions:
            for i, trans in enumerate(st.session_state.new_transactions[:]):
                with st.expander(
                        f"Transaction {i + 1}: {trans.get('type', 'N/A')} - ${trans.get('amount', 0):,.2f} | Status: {trans.get('status', 'N/A')} | Fraud: {'Yes' if trans.get('is_fraud', False) else 'No'}"):
                    cols = st.columns(2)
                    with cols[0]:
                        st.write("### Transaction Details")
                        st.write(f"Time: {trans.get('timestamp', 'N/A')}")
                        st.write(f"From: {trans.get('customer_id', 'N/A')}")
                        st.write(f"To: {trans.get('dest_id', 'N/A')}")
                        st.write(f"Type: {trans.get('type', 'N/A')}")
                        st.write(f"Amount: ${trans.get('amount', 0):,.2f}")
                        st.write(f"Description: {trans.get('description', 'N/A')}")

                    with cols[1]:
                        st.write("### Risk Analysis")
                        st.write(f"Risk Score: {trans.get('risk_score', 0) * 100:.1f}%")
                        st.write(f"Fraud Detected: {'Yes' if trans.get('is_fraud', False) else 'No'}")
                        st.write(
                            f"Risk Factors: {', '.join([*trans.get('business_rules', []), *trans.get('kyc_rules', [])])}")

                        # Show detection details
                        st.write("### Detection Details")
                        predictions = trans.get('predictions', {})
                        st.write(f"- Supervised Model: {'Fraud' if predictions.get('supervised', 0) else 'Clean'}")
                        st.write(f"- MLP Model: {'Fraud' if predictions.get('mlp', 0) else 'Clean'}")
                        st.write(
                            f"- Unsupervised Model: {'Anomaly' if predictions.get('unsupervised', 0) else 'Normal'}")
                        st.write(
                            f"- Graph Analysis: {'Suspicious' if predictions.get('graph_anomaly', False) else 'Normal'}")
                        st.write(f"- Time Series: {'Anomaly' if predictions.get('time_anomaly', False) else 'Normal'}")

                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"Approve Transaction {i + 1}"):
                            trans['status'] = 'Approved'
                            if trans.get('is_fraud', False):
                                st.session_state.admin_metrics['true_positives'] += 1
                            else:
                                update_balances(
                                    trans['customer_id'],
                                    trans['dest_id'],
                                    trans['amount'],
                                    trans['type']
                                )
                            st.session_state.new_transactions.remove(trans)
                            st.session_state.admin_metrics['active_alerts'] = max(
                                0, st.session_state.admin_metrics['active_alerts'] - 1)
                            st.rerun()
                    with col2:
                        if st.button(f"Reject Transaction {i + 1}"):
                            trans['status'] = 'Rejected'
                            st.session_state.new_transactions.remove(trans)
                            st.session_state.admin_metrics['active_alerts'] = max(
                                0, st.session_state.admin_metrics['active_alerts'] - 1)
                            st.rerun()
                    with col3:
                        if st.button(f"View Details {i + 1}"):
                            st.json(trans)
        else:
            st.info("No new transactions awaiting review")

        # Fraud alerts section
        st.subheader("Fraud Alerts")
        if st.session_state.fraud_alerts:
            for i, alert in enumerate(st.session_state.fraud_alerts[:]):
                with st.expander(
                        f"🚨 Alert {i + 1}: {alert.get('type', 'N/A')} - ${alert.get('amount', 0):,.2f} | Risk: {alert.get('risk_score', 0) * 100:.1f}% | Confirmed: {'Yes' if alert.get('confirmed', False) else 'No'}"):
                    cols = st.columns(2)
                    with cols[0]:
                        st.write("### Transaction Details")
                        st.write(f"Time: {alert.get('timestamp', 'N/A')}")
                        st.write(f"From: {alert.get('customer_id', 'N/A')}")
                        st.write(f"To: {alert.get('dest_id', 'N/A')}")
                        st.write(f"Type: {alert.get('type', 'N/A')}")
                        st.write(f"Amount: ${alert.get('amount', 0):,.2f}")

                    with cols[1]:
                        st.write("### Fraud Analysis")
                        st.write(f"Risk Score: {alert.get('risk_score', 0) * 100:.1f}%")
                        st.write(
                            f"Risk Factors: {', '.join([*alert.get('business_rules', []), *alert.get('kyc_rules', [])])}")
                        st.write(f"Model Predictions:")
                        st.write(
                            f"- Supervised: {'Fraud' if alert.get('predictions', {}).get('supervised', 0) else 'Clean'}")
                        st.write(f"- MLP: {'Fraud' if alert.get('predictions', {}).get('mlp', 0) else 'Clean'}")
                        st.write(
                            f"- Unsupervised: {'Anomaly' if alert.get('predictions', {}).get('unsupervised', 0) else 'Normal'}")
                        st.write(
                            f"- Graph Analysis: {'Suspicious' if alert.get('predictions', {}).get('graph_anomaly', False) else 'Normal'}")
                        st.write(
                            f"- Time Series: {'Anomaly' if alert.get('predictions', {}).get('time_anomaly', False) else 'Normal'}")

                    # Action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Confirm Fraud {i + 1}"):
                            alert['confirmed'] = True
                            st.session_state.admin_metrics['true_positives'] += 1
                            st.rerun()
                    with col2:
                        if st.button(f"False Alarm {i + 1}"):
                            alert['confirmed'] = False
                            update_balances(
                                alert['customer_id'],
                                alert['dest_id'],
                                alert['amount'],
                                alert['type']
                            )
                            st.rerun()
        else:
            st.info("No active fraud alerts")

    except Exception as e:
        st.error(f"Error in admin panel: {str(e)}")


# Main App
def main():
    st.set_page_config(page_title="Fraud Detection System", layout="wide")

    # Initialize session state
    initialize_session_state()

    # Load data and models
    df, df_for_lookup = load_dataset()
    models = initialize_models(df) if not df.empty else None

    # Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Mode", ["User Panel", "Admin Panel"])

    if app_mode == "User Panel":
        display_user_panel(df_for_lookup, models)
    else:
        display_admin_panel()


if __name__ == "__main__":
    main()
