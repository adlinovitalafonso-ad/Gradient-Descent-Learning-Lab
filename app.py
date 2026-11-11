import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import sqlite3, json
from datetime import datetime

st.set_page_config(page_title='Optimizer Learning Lab', layout='wide')

# ---------- Access codes ----------
def secret(name, fallback):
    try:
        return st.secrets[name]
    except Exception:
        return fallback

STUDENT_CODE = secret('STUDENT_ACCESS_CODE', 'MMAIO2026')
FACULTY_CODE = secret('FACULTY_ACCESS_CODE', 'FACULTY2026')
DB='optimizer_lab.db'

# ---------- Database ----------
def conn(): return sqlite3.connect(DB, check_same_thread=False)
def init_db():
    c=conn(); q=c.cursor()
    q.execute('''CREATE TABLE IF NOT EXISTS students(
      email TEXT, roll_no TEXT, first_login TEXT, last_login TEXT,
      UNIQUE(email,roll_no))''')
    q.execute('''CREATE TABLE IF NOT EXISTS activity(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      email TEXT, roll_no TEXT, ts TEXT, problem_type TEXT,
      optimizer TEXT, action TEXT, learning_rate REAL,
      iteration INTEGER, loss REAL, parameters TEXT,
      converged INTEGER, stopping_reason TEXT, extra TEXT)''')
    c.commit(); c.close()
init_db()

def upsert_student(email, roll):
    now=datetime.now().isoformat(timespec='seconds'); c=conn(); q=c.cursor()
    q.execute('''INSERT INTO students VALUES(?,?,?,?)
      ON CONFLICT(email,roll_no) DO UPDATE SET last_login=excluded.last_login''',
      (email,roll,now,now)); c.commit(); c.close()

def log(problem, opt, action, lr=None, it=None, loss=None, params=None,
        converged=False, reason='', extra=None):
    if not st.session_state.get('logged_in'): return
    c=conn(); q=c.cursor(); q.execute('''INSERT INTO activity(
      email,roll_no,ts,problem_type,optimizer,action,learning_rate,
      iteration,loss,parameters,converged,stopping_reason,extra)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',(
      st.session_state.email, st.session_state.roll_no,
      datetime.now().isoformat(timespec='seconds'), problem,opt,action,lr,it,loss,
      json.dumps(params) if params is not None else None,
      int(converged), reason,
      json.dumps(extra) if extra is not None else None)); c.commit(); c.close()

def df_sql(sql, params=()):
    c=conn(); d=pd.read_sql_query(sql,c,params=params); c.close(); return d

# ---------- Optimizers ----------
def update(params, grad, lr, opt, state, beta=.9, b1=.9, b2=.999, eps=1e-8):
    p=np.array(params,float); g=np.array(grad,float)
    if opt in ['Gradient Descent','Batch Gradient Descent','SGD']:
        return p-lr*g, state
    if opt=='Momentum':
        state.setdefault('v',np.zeros_like(p)); state['v']=beta*state['v']+g
        return p-lr*state['v'], state
    state.setdefault('m',np.zeros_like(p)); state.setdefault('v',np.zeros_like(p)); state.setdefault('t',0)
    state['t']+=1; state['m']=b1*state['m']+(1-b1)*g; state['v']=b2*state['v']+(1-b2)*(g*g)
    mh=state['m']/(1-b1**state['t']); vh=state['v']/(1-b2**state['t'])
    return p-lr*mh/(np.sqrt(vh)+eps), state

def reset(prefix):
    for k in list(st.session_state.keys()):
        if k.startswith(prefix): del st.session_state[k]

def gnorm(g): return float(np.linalg.norm(np.array(g,float)))

# ---------- Session defaults ----------
st.session_state.setdefault('logged_in',False)
st.session_state.setdefault('faculty_mode',False)
st.session_state.setdefault('show_history',False)

# ---------- Login ----------
if not st.session_state.logged_in and not st.session_state.faculty_mode:
    st.title('Optimizer Learning Lab')
    st.subheader('Student Login')
    st.caption('Email ID and Roll Number are recorded for course participation and learning analytics.')
    with st.form('login'):
        email=st.text_input('Email ID *'); roll=st.text_input('Roll Number *'); code=st.text_input('Access Code *',type='password')
        go=st.form_submit_button('Enter Lab')
    if go:
        if not email.strip() or not roll.strip() or not code.strip(): st.error('All three fields are mandatory.')
        elif '@' not in email: st.error('Enter a valid email ID.')
        elif code.strip()!=STUDENT_CODE: st.error('Incorrect access code.')
        else:
            st.session_state.logged_in=True; st.session_state.email=email.strip(); st.session_state.roll_no=roll.strip()
            upsert_student(st.session_state.email,st.session_state.roll_no); log('Login','','Student Login'); st.rerun()
    st.divider()
    with st.expander('Faculty Login'):
        with st.form('faculty'):
            fc=st.text_input('Faculty Access Code',type='password'); fg=st.form_submit_button('Open Analytics')
        if fg:
            if fc==FACULTY_CODE: st.session_state.faculty_mode=True; st.rerun()
            else: st.error('Incorrect faculty code.')
    st.stop()

# ---------- Faculty analytics ----------
if st.session_state.faculty_mode:
    st.title('Faculty Analytics Dashboard')
    if st.button('Exit Faculty Mode'): st.session_state.faculty_mode=False; st.rerun()
    students=df_sql('SELECT * FROM students ORDER BY last_login DESC')
    activity=df_sql('SELECT * FROM activity ORDER BY id DESC')
    a,b,c,d=st.columns(4)
    a.metric('Unique Students',len(students)); b.metric('Logged Actions',len(activity))
    c.metric('Optimization Actions',int(activity['action'].isin(['Next Step','Run 4 Steps','Run to Convergence']).sum()) if len(activity) else 0)
    d.metric('Converged Runs',int(activity['converged'].sum()) if len(activity) else 0)
    tabs=st.tabs(['Overview','Students','Optimizers','Learning Rates','Activity Log','Export'])
    with tabs[0]:
        if len(activity):
            x=activity[activity.problem_type!='Login'].problem_type.value_counts();
            if len(x):
                fig,ax=plt.subplots(); x.plot(kind='bar',ax=ax); ax.set_ylabel('Actions'); st.pyplot(fig)
            st.dataframe(activity.head(30),use_container_width=True)
        else: st.info('No activity yet.')
    with tabs[1]:
        st.dataframe(students,use_container_width=True)
        if len(activity):
            s=activity.groupby(['email','roll_no']).agg(actions=('id','count'),optimizers=('optimizer','nunique'),convergence_events=('converged','sum'),last_activity=('ts','max')).reset_index()
            st.dataframe(s,use_container_width=True)
    with tabs[2]:
        if len(activity):
            x=activity[activity.optimizer.fillna('')!=''].optimizer.value_counts();
            if len(x):
                fig,ax=plt.subplots(); x.plot(kind='bar',ax=ax); ax.set_ylabel('Actions'); st.pyplot(fig)
                st.dataframe(x.rename('Actions').reset_index(),use_container_width=True)
    with tabs[3]:
        x=activity[activity.learning_rate.notna()] if len(activity) else pd.DataFrame()
        st.dataframe(x[['email','roll_no','problem_type','optimizer','learning_rate','action','loss','converged']] if len(x) else x,use_container_width=True)
    with tabs[4]: st.dataframe(activity,use_container_width=True)
    with tabs[5]:
        if len(activity): st.download_button('Download Activity CSV',activity.to_csv(index=False).encode(),'optimizer_activity.csv','text/csv')
        if len(students): st.download_button('Download Student CSV',students.to_csv(index=False).encode(),'optimizer_students.csv','text/csv')
    st.stop()

# ---------- Student header ----------
st.title('Optimizer Learning Lab')
c1,c2,c3=st.columns([2,1,1])
with c1: st.caption(f'Logged in as {st.session_state.email} | Roll No: {st.session_state.roll_no}')
with c2:
    if st.button('My Experiments'): st.session_state.show_history=not st.session_state.show_history; st.rerun()
with c3:
    if st.button('Logout'): st.session_state.logged_in=False; st.rerun()
if st.session_state.show_history:
    h=df_sql('''SELECT ts,problem_type,optimizer,action,learning_rate,iteration,loss,parameters,converged,stopping_reason
                FROM activity WHERE email=? AND roll_no=? ORDER BY id DESC''',(st.session_state.email,st.session_state.roll_no))
    st.subheader('My Experiment History'); st.dataframe(h,use_container_width=True); st.divider()

mode=st.sidebar.radio('Problem Type',['1D Function','2D Function','Linear Regression'])
st.sidebar.caption('Use Next Step to inspect one move, Run 4 Steps for demonstration, and Run to Convergence for the final result.')

# ==================== 1D ====================
if mode=='1D Function':
    st.header('1D Function Optimization')
    left,right=st.columns(2)
    with left:
        expression=st.text_input('Enter f(x)', '(x-4)**2')
        x0=st.number_input('Starting x₀',value=0.0); opt=st.selectbox('Optimizer',['Gradient Descent','Momentum','Adam'])
        lr=st.number_input('Learning rate α',min_value=1e-6,value=.1,format='%.6f'); tol=st.number_input('Gradient tolerance ε',min_value=1e-7,value=.01,format='%.7f'); maxit=st.number_input('Maximum iterations',1,500,500)
        beta=.9; b1=.9; b2=.999
        if opt=='Momentum': beta=st.slider('Momentum β',0.0,.999,.9,.01)
        if opt=='Adam': b1=st.slider('Adam β₁',0.0,.999,.9,.01); b2=st.slider('Adam β₂',0.0,.9999,.999,.0001)
    x=sp.symbols('x')
    try:
        expr=sp.sympify(expression); dexpr=sp.diff(expr,x); f=sp.lambdify(x,expr,'numpy'); df=sp.lambdify(x,dexpr,'numpy')
        st.latex('f(x)='+sp.latex(expr)); st.latex("f'(x)="+sp.latex(dexpr))
    except Exception as e: st.error(str(e)); st.stop()
    pre='one_'
    if pre+'init' not in st.session_state:
        st.session_state[pre+'init']=1; st.session_state[pre+'x']=float(x0); st.session_state[pre+'it']=0; st.session_state[pre+'hist']=[]; st.session_state[pre+'state']={}
    def step1():
        xo=float(st.session_state[pre+'x']); it=st.session_state[pre+'it']; g=float(df(xo)); pn,s=update([xo],[g],lr,opt,st.session_state[pre+'state'],beta,b1,b2); xn=float(pn[0]); fx=float(f(xn))
        st.session_state[pre+'hist'].append({'Iteration':it,'x_old':xo,'Gradient':g,'Step':xn-xo,'x_new':xn,'f(x_new)':fx}); st.session_state[pre+'x']=xn; st.session_state[pre+'it']=it+1; st.session_state[pre+'state']=s
    buttons=st.columns(4)
    with buttons[0]:
        if st.button('Reset',key='r1',use_container_width=True): reset(pre); st.rerun()
    with buttons[1]:
        if st.button('Next Step',key='n1',use_container_width=True):
            step1(); xc=float(st.session_state[pre+'x']); conv=abs(float(df(xc)))<tol; log('1D Function',opt,'Next Step',lr,st.session_state[pre+'it'],float(f(xc)),[xc],conv,'Gradient tolerance reached' if conv else ''); st.rerun()
    with buttons[2]:
        if st.button('Run 4 Steps',key='f1',use_container_width=True):
            for _ in range(4): step1()
            xc=float(st.session_state[pre+'x']); conv=abs(float(df(xc)))<tol; log('1D Function',opt,'Run 4 Steps',lr,st.session_state[pre+'it'],float(f(xc)),[xc],conv); st.rerun()
    with buttons[3]:
        if st.button('Run to Convergence',key='c1',use_container_width=True):
            reason='Maximum iterations reached'
            for _ in range(int(maxit)):
                if abs(float(df(st.session_state[pre+'x'])))<tol: reason='Gradient tolerance reached'; break
                step1()
            xc=float(st.session_state[pre+'x']); conv=abs(float(df(xc)))<tol; log('1D Function',opt,'Run to Convergence',lr,st.session_state[pre+'it'],float(f(xc)),[xc],conv,reason); st.rerun()
    xc=float(st.session_state[pre+'x']); gc=float(df(xc)); fx=float(f(xc)); m=st.columns(4); m[0].metric('Iteration',st.session_state[pre+'it']); m[1].metric('Current x',f'{xc:.6f}'); m[2].metric('Gradient',f'{gc:.6f}'); m[3].metric('Function value',f'{fx:.6f}')
    if abs(gc)<tol: st.success(f'Converged: |gradient|={abs(gc):.6f} < ε={tol}')
    hist=pd.DataFrame(st.session_state[pre+'hist']);
    if len(hist): st.dataframe(hist.tail(20),use_container_width=True)
    with right:
        st.subheader('Optimization Path'); xs=[float(x0)] if not len(hist) else [hist.iloc[0].x_old]+hist.x_new.tolist(); xx=np.linspace(min(xs)-2,max(xs)+2,400); yy=f(xx); fig,ax=plt.subplots(); ax.plot(xx,yy,label='f(x)')
        if len(hist): py=[float(f(v)) for v in xs]; ax.plot(xs,py,'o--',label='Steps'); [ax.annotate(str(i),(px,pyy)) for i,(px,pyy) in enumerate(zip(xs[:5],py[:5]))]
        ax.legend(); ax.grid(alpha=.2); st.pyplot(fig)

# ==================== 2D ====================
elif mode=='2D Function':
    st.header('2D Quadratic Optimization'); st.latex(r'f(x_1,x_2)=ax_1^2+bx_2^2+cx_1x_2+dx_1+ex_2+k')
    c=st.columns(3); a=c[0].number_input('a',value=1.0); d=c[0].number_input('d',value=0.0); b=c[1].number_input('b',value=1.0); e=c[1].number_input('e',value=0.0); cc=c[2].number_input('c',value=0.0); k=c[2].number_input('k',value=0.0)
    c=st.columns(2); x10=c[0].number_input('Starting x₁',value=4.0); x20=c[0].number_input('Starting x₂',value=2.0); opt=c[0].selectbox('Optimizer',['Gradient Descent','Momentum','Adam'],key='o2'); lr=c[1].number_input('Learning rate α',min_value=1e-6,value=.1,key='lr2'); tol=c[1].number_input('Gradient norm tolerance ε',min_value=1e-7,value=.01,key='t2'); maxit=c[1].number_input('Maximum iterations',1,500,500,key='m2')
    beta=.9; b1=.9; b2=.999
    if opt=='Momentum': beta=st.slider('Momentum β',0.0,.999,.9,.01,key='mb2')
    if opt=='Adam': b1=st.slider('Adam β₁',0.0,.999,.9,.01,key='ab12'); b2=st.slider('Adam β₂',0.0,.9999,.999,.0001,key='ab22')
    def f2(p): x1,x2=p; return a*x1*x1+b*x2*x2+cc*x1*x2+d*x1+e*x2+k
    def g2(p): x1,x2=p; return np.array([2*a*x1+cc*x2+d,2*b*x2+cc*x1+e],float)
    eig=np.linalg.eigvalsh(np.array([[2*a,cc],[cc,2*b]],float)); st.info('Strictly convex: stationary minimum is global.' if np.all(eig>0) else 'This setting is not strictly convex; the path may not lead to a unique global minimum.')
    pre='two_'
    if pre+'init' not in st.session_state: st.session_state[pre+'init']=1; st.session_state[pre+'p']=np.array([x10,x20],float); st.session_state[pre+'it']=0; st.session_state[pre+'hist']=[]; st.session_state[pre+'state']={}
    def step2():
        p=np.array(st.session_state[pre+'p'],float); it=st.session_state[pre+'it']; g=g2(p); pn,s=update(p,g,lr,opt,st.session_state[pre+'state'],beta,b1,b2); st.session_state[pre+'hist'].append({'Iteration':it,'x1_old':p[0],'x2_old':p[1],'grad_x1':g[0],'grad_x2':g[1],'x1_new':pn[0],'x2_new':pn[1],'f(new)':f2(pn)}); st.session_state[pre+'p']=pn; st.session_state[pre+'it']=it+1; st.session_state[pre+'state']=s
    bs=st.columns(4)
    if bs[0].button('Reset',key='r2',use_container_width=True): reset(pre); st.rerun()
    if bs[1].button('Next Step',key='n2',use_container_width=True): step2(); p=st.session_state[pre+'p']; conv=gnorm(g2(p))<tol; log('2D Function',opt,'Next Step',lr,st.session_state[pre+'it'],f2(p),p.tolist(),conv); st.rerun()
    if bs[2].button('Run 4 Steps',key='f2',use_container_width=True):
        for _ in range(4): step2()
        p=st.session_state[pre+'p']; conv=gnorm(g2(p))<tol; log('2D Function',opt,'Run 4 Steps',lr,st.session_state[pre+'it'],f2(p),p.tolist(),conv); st.rerun()
    if bs[3].button('Run to Convergence',key='c2',use_container_width=True):
        reason='Maximum iterations reached'
        for _ in range(int(maxit)):
            if gnorm(g2(st.session_state[pre+'p']))<tol: reason='Gradient tolerance reached'; break
            step2()
        p=st.session_state[pre+'p']; conv=gnorm(g2(p))<tol; log('2D Function',opt,'Run to Convergence',lr,st.session_state[pre+'it'],f2(p),p.tolist(),conv,reason); st.rerun()
    p=np.array(st.session_state[pre+'p']); m=st.columns(4); m[0].metric('Iteration',st.session_state[pre+'it']); m[1].metric('x₁',f'{p[0]:.5f}'); m[2].metric('x₂',f'{p[1]:.5f}'); m[3].metric('||Gradient||',f'{gnorm(g2(p)):.6f}')
    hist=pd.DataFrame(st.session_state[pre+'hist']);
    if len(hist): st.dataframe(hist.tail(20),use_container_width=True)
    path=np.array([[x10,x20]]) if not len(hist) else np.vstack([[hist.iloc[0].x1_old,hist.iloc[0].x2_old],hist[['x1_new','x2_new']].values]); X1,X2=np.meshgrid(np.linspace(path[:,0].min()-2,path[:,0].max()+2,150),np.linspace(path[:,1].min()-2,path[:,1].max()+2,150)); Z=a*X1**2+b*X2**2+cc*X1*X2+d*X1+e*X2+k; fig,ax=plt.subplots(); ax.contour(X1,X2,Z,20); ax.plot(path[:,0],path[:,1],'o--'); [ax.annotate(str(i),(px,py)) for i,(px,py) in enumerate(path[:5])]; ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); st.pyplot(fig)

# ==================== Regression ====================
else:
    st.header('Linear Regression Optimization')
    source=st.radio('Data Source',['Manual input','Upload CSV'],horizontal=True)
    if source=='Manual input':
        data=st.data_editor(pd.DataFrame({'x':[1.,2.,3.,4.,5.],'y':[3.,5.,7.,9.,11.]}),num_rows='dynamic',use_container_width=True); xcol='x'; ycol='y'
    else:
        up=st.file_uploader('Upload CSV',type='csv');
        if up is None: st.stop()
        raw=pd.read_csv(up); nums=raw.select_dtypes(include=np.number).columns.tolist();
        if len(nums)<2: st.error('Need at least two numeric columns.'); st.stop()
        c=st.columns(2); xcol=c[0].selectbox('X column',nums); ycol=c[1].selectbox('Y column',nums,index=1); data=raw[[xcol,ycol]].copy(); data.columns=['x','y']
    data=data[['x','y']].dropna(); X=data.x.astype(float).to_numpy(); Y=data.y.astype(float).to_numpy(); n=len(X)
    c=st.columns(3); b0=c[0].number_input('Initial intercept b₀',value=0.0); b1s=c[0].number_input('Initial slope b₁',value=0.0); opt=c[1].selectbox('Optimizer',['Batch Gradient Descent','SGD','Momentum','Adam'],key='or'); lr=c[1].number_input('Learning rate α',min_value=1e-7,value=.01,key='lrr'); tol=c[2].number_input('Gradient tolerance ε',min_value=1e-7,value=.001,key='tr'); maxit=c[2].number_input('Maximum iterations',1,5000,1000,key='mr')
    beta=.9; ab1=.9; ab2=.999
    if opt=='Momentum': beta=st.slider('Momentum β',0.0,.999,.9,.01,key='mbr')
    if opt=='Adam': ab1=st.slider('Adam β₁',0.0,.999,.9,.01,key='a1r'); ab2=st.slider('Adam β₂',0.0,.9999,.999,.0001,key='a2r')
    def mse(p): pr=p[0]+p[1]*X; return float(np.mean((Y-pr)**2))
    def bg(p): pr=p[0]+p[1]*X; er=Y-pr; return np.array([-2*np.mean(er),-2*np.mean(X*er)])
    def sg(p,i): er=Y[i]-(p[0]+p[1]*X[i]); return np.array([-2*er,-2*X[i]*er])
    pre='reg_'
    if pre+'init' not in st.session_state: st.session_state[pre+'init']=1; st.session_state[pre+'p']=np.array([b0,b1s],float); st.session_state[pre+'it']=0; st.session_state[pre+'hist']=[]; st.session_state[pre+'state']={}; st.session_state[pre+'idx']=0
    def stepr():
        p=np.array(st.session_state[pre+'p'],float); it=st.session_state[pre+'it']; used='All'
        if opt=='SGD': i=st.session_state[pre+'idx']%n; g=sg(p,i); used=i+1; st.session_state[pre+'idx']=i+1
        else: g=bg(p)
        pn,s=update(p,g,lr,opt,st.session_state[pre+'state'],beta,ab1,ab2); st.session_state[pre+'hist'].append({'Iteration':it,'Data used':used,'b0_old':p[0],'b1_old':p[1],'grad_b0':g[0],'grad_b1':g[1],'b0_new':pn[0],'b1_new':pn[1],'MSE':mse(pn)}); st.session_state[pre+'p']=pn; st.session_state[pre+'it']=it+1; st.session_state[pre+'state']=s
    bs=st.columns(4)
    if bs[0].button('Reset',key='rr',use_container_width=True): reset(pre); st.rerun()
    if bs[1].button('Next Step',key='nr',use_container_width=True): stepr(); p=st.session_state[pre+'p']; conv=gnorm(bg(p))<tol; log('Linear Regression',opt,'Next Step',lr,st.session_state[pre+'it'],mse(p),p.tolist(),conv,extra={'rows':n}); st.rerun()
    if bs[2].button('Run 4 Steps',key='fr',use_container_width=True):
        for _ in range(4): stepr()
        p=st.session_state[pre+'p']; conv=gnorm(bg(p))<tol; log('Linear Regression',opt,'Run 4 Steps',lr,st.session_state[pre+'it'],mse(p),p.tolist(),conv,extra={'rows':n}); st.rerun()
    if bs[3].button('Run to Convergence',key='cr',use_container_width=True):
        reason='Maximum iterations reached'
        for _ in range(int(maxit)):
            if gnorm(bg(st.session_state[pre+'p']))<tol: reason='Gradient tolerance reached'; break
            stepr()
        p=st.session_state[pre+'p']; conv=gnorm(bg(p))<tol; log('Linear Regression',opt,'Run to Convergence',lr,st.session_state[pre+'it'],mse(p),p.tolist(),conv,reason,{'rows':n}); st.rerun()
    p=np.array(st.session_state[pre+'p']); m=st.columns(4); m[0].metric('Iteration',st.session_state[pre+'it']); m[1].metric('Intercept b₀',f'{p[0]:.5f}'); m[2].metric('Slope b₁',f'{p[1]:.5f}'); m[3].metric('MSE',f'{mse(p):.6f}')
    hist=pd.DataFrame(st.session_state[pre+'hist']);
    if len(hist): st.dataframe(hist.tail(20),use_container_width=True)
    l,r=st.columns(2)
    with l:
        fig,ax=plt.subplots(); ax.scatter(X,Y,label='Observed'); xx=np.linspace(X.min(),X.max(),100)
        if len(hist):
            for _,row in hist.head(4).iterrows(): ax.plot(xx,row.b0_new+row.b1_new*xx,alpha=.35)
        ax.plot(xx,p[0]+p[1]*xx,linewidth=2.5,label=f'Current: y={p[0]:.3f}+{p[1]:.3f}x'); ax.legend(); ax.grid(alpha=.2); st.pyplot(fig)
    with r:
        fig,ax=plt.subplots();
        if len(hist): ax.plot(hist.Iteration+1,hist.MSE,'o-')
        ax.set_xlabel('Iteration'); ax.set_ylabel('MSE'); ax.grid(alpha=.2); st.pyplot(fig)
    st.subheader('Teaching Interpretation')
    msg={'Batch Gradient Descent':'Uses the whole dataset for each update.','SGD':'Uses one observation per update, so the path is noisier.','Momentum':'Adds memory of previous movement to the current gradient.','Adam':'Combines momentum-like memory with adaptive step sizes.'}[opt]
    st.write(msg)
