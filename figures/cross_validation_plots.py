

i = 0
fig, ax = plt.subplots(1, 2, figsize=(6, 6))  # setup the plot
for i in np.unique(seg_base[cluster_var]):
    scat1 = ax[0].scatter(seg_base.loc[seg_base[cluster_var] == i]["lon"],
                          seg_base.loc[seg_base[cluster_var] == i]["lat"], label=i)  # , s=date_norm
    scat1 = ax[1].scatter(seg_base.loc[seg_base[cluster_var] == i]["lon"],
                          seg_base.loc[seg_base[cluster_var] == i]["lat"], label=i)
ax[0].set_xlim([np.min(seg_base["lon"])-2, -100])
ax[0].set_ylim([30, np.max(seg_base["lat"])+2])

ax[1].set_xlim([115, np.max(seg_base["lon"])+1])
ax[1].set_ylim([np.min(seg_base["lat"])-2, -10])

fig.text(0.5, 0.04, 'longitude', ha='center')
fig.text(0.04, 0.5, 'latitude', va='center', rotation='vertical')

ax[0].set_title('North America')
ax[1].set_title('Australia')
ax[1].legend(bbox_to_anchor=(1.32, 1.0))
#plt.savefig(fileRes, format='png', dpi=300)
