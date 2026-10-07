import numpy as np
from matplotlib.patches import Rectangle

PALETTE={'background':'#a7b5c9','connected':'#41e2bc','hidden':'#faab62','road':'#27334a'}

def draw_scene(ax,road,state,connected=None,visible=None,title='',trajectories=None):
    ax.set_facecolor('#101827');ax.set_aspect('equal');ax.set_xlim(-85,85);ax.set_ylim(-85,85)
    for lane in road.lanes:
        p=lane.centerline
        ax.plot(p[:,0],p[:,1],color=PALETTE['road'],lw=11,zorder=1)
        ax.plot(p[:,0],p[:,1],color='#71819a',lw=.6,ls='--',alpha=.7,zorder=2)
    if trajectories is not None:
        for i in range(trajectories.shape[1]):
            ax.plot(trajectories[:,i,0]*100,trajectories[:,i,1]*100,color=PALETTE['connected'] if connected is not None and connected[i] else PALETTE['background'],alpha=.7,lw=1)
    for i,s in enumerate(state):
        if s[5]<.2: continue
        color=PALETTE['connected'] if connected is not None and connected[i] else PALETTE['background']
        if visible is not None and not visible[i]: color=PALETTE['hidden']
        x,y=s[:2]*100
        ax.scatter(x,y,s=45 if connected is not None and connected[i] else 30,c=color,marker='s',edgecolors='#101827',linewidths=.5,zorder=3)
        ax.text(x+2,y+2,str(i),color='white',fontsize=6,zorder=4)
    ax.set_title(title,color='#e4ecf9',fontsize=10,pad=9)
    ax.tick_params(colors='#667b97',labelsize=6)
    for spine in ax.spines.values(): spine.set_color('#34435b')
