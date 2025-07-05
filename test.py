#!/usr/bin/env python
# coding: utf-8

# # PSMILES

# In[2]:

# In[21]:


from psmiles import PolymerSmiles as PS
ps = PS('[*]C(=O)OCCOC([*])N')
ps


# In[8]:


# Canonicalize the PSMILES string.
ps.canonicalize


# In[9]:


# Save the figure to disk. Default name to PSMILES string.
ps.canonicalize.savefig()


# In[10]:



# Get the dimer from the monomer. 
# Connect to first star
ps.dimer(0)


# In[ ]:


# Connect to second star
ps.dimer(1)


# ### Fingerprints for PSMILES strings

# In[ ]:


# CI fingerprint
ps.fingerprint('ci')


# In[ ]: