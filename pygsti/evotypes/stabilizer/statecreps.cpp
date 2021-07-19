#define NULL 0

#include <iostream>
#include <complex>
#include <assert.h>
#include <algorithm>    // std::find
#include "statecreps.h"
//#include <pthread.h>

//using namespace std::complex_literals;

//#define DEBUG(x) x
#define DEBUG(x) 

namespace CReps_stabilizer {

  /****************************************************************************\
  |* StateCRep                                                              *|
  \****************************************************************************/

  StateCRep::StateCRep(INT* smatrix, INT* pvectors, dcomplex* amps, INT namps, INT n) {
    _smatrix = smatrix;
    _pvectors = pvectors;
    _amps = amps;
    _namps = namps;
    _n = n;
    _2n = 2*n;
    _ownmem = false;
    rref(); // initializes _zblock_start
  }

  StateCRep::StateCRep(INT namps, INT n) {
    _n = n;
    _2n = 2*n;
    _namps = namps;
    _smatrix = new INT[_2n*_2n];
    _pvectors = new INT[_namps*_2n];
    _amps = new dcomplex[_namps];
    _ownmem = true;
    _zblock_start = -1;
  }
  
  StateCRep::~StateCRep() {
    if(_ownmem) {
      delete [] _smatrix;
      delete [] _pvectors;
      delete [] _amps;
    }
  }
  
  void StateCRep::push_view(std::vector<INT>& view) {
    _view_filters.push_back(view);
  }
  
  void StateCRep::pop_view() {
    _view_filters.pop_back();
  }
    
  void StateCRep::clifford_update(INT* smatrix, INT* svector, dcomplex* Umx) {
    //vs = (_np.array([1,0],complex), _np.array([0,1],complex)) # (v0,v1)
    DEBUG(std::cout << "Clifford Update BEGIN" << std::endl);
    INT i,k,ip;
    std::vector<std::vector<INT> >::iterator it;
    std::vector<INT>::iterator it2;

    std::vector<INT> qubits(_n);
    for(i=0; i<_n; i++) qubits[i] = i; // start with all qubits being acted on
    for(it=_view_filters.begin(); it != _view_filters.end(); ++it) {
      std::vector<INT>& qfilter = *it;
      std::vector<INT> new_qubits(qfilter.size());
      for(i=0; i < (INT)qfilter.size(); i++)
	new_qubits[i] = qubits[ qfilter[i] ]; // apply each filter
      qubits.resize( new_qubits.size() );
      for(i=0; i < (INT)qfilter.size(); i++)
	qubits[i] = new_qubits[i]; //copy new_qubits -> qubits (maybe a faster way?)
    }

    INT nQ = qubits.size(); //number of qubits being acted on (<= n in general)
    std::vector<std::vector<INT> > sampled_states(_namps);
    std::vector<std::vector<dcomplex> > sampled_amplitudes(_namps);

    INT action_size = pow(2,qubits.size());
    std::vector<dcomplex> outstate(action_size);

    // Step1: Update global amplitudes - Part A
    DEBUG(std::cout << "UPDATE GLOBAL AMPS: zstart=" << _zblock_start << std::endl);
    for(ip=0; ip<_namps; ip++) {
      sampled_states[ip].resize(_n);
      sampled_amplitudes[ip].resize( action_size );
      canonical_amplitudes_sample(ip,qubits, sampled_states[ip], sampled_amplitudes[ip]);
    }
    
    // Step2: Apply clifford to stabilizer reps in _smatrix, _pvectors
    DEBUG(std::cout << "APPLY CLIFFORD TO FRAME" << std::endl);
    apply_clifford_to_frame(smatrix, svector, qubits);
    rref();

    //DEBUG!!! - print smx and pvecs
    //std::cout << "S = ";
    //for(i=0; i<_2n*_2n; i++) std::cout << _smatrix[i] << " ";
    //std::cout << std::endl;
    //std::cout << "PS = ";
    //for(i=0; i<_namps*_2n; i++) std::cout << _pvectors[i] << " ";
    //std::cout << std::endl;

    // Step3: Update global amplitudes - Part B
    for(ip=0; ip<_namps; ip++) {
      const std::vector<INT> & base_state = sampled_states[ip];
      const std::vector<dcomplex> & ampls = sampled_amplitudes[ip];

      //DEBUG!!! - print Umx
      //std::cout << "U = ";
      //for(i=0; i<action_size*action_size; i++) std::cout << Umx[i] << " ";
      //std::cout << std::endl;

      // APPLYING U to instate = ampls, i.e. outstate = _np.dot(Umx,ampls)
      DEBUG(std::cout << "APPLYING U to instate = ");
      DEBUG(for(i=0; i<action_size; i++) std::cout << ampls[i] << " ");
      DEBUG(std::cout << std::endl);
      for(i=0; i<action_size; i++) {
	outstate[i] = 0.0;
	for(k=0; k<action_size; k++)
	  outstate[i] += Umx[i*action_size+k] * ampls[k]; // state-vector propagation
      }
      DEBUG(std::cout << "outstate = ");
      DEBUG(for(i=0; i<action_size; i++) std::cout << outstate[i] << " ");
      DEBUG(std::cout << std::endl);


      //Look for nonzero output component and figure out how
      // phase *actually* changed as per state-vector propagation, then
      // update self.a (global amplitudes) to account for this.
      for(k=0; k<action_size; k++) {
	dcomplex comp = outstate[k]; // component of output state
	if(std::abs(comp) > 1e-6) {
	  std::vector<INT> zvals(base_state);
	  std::vector<INT> k_zvals(nQ);
	  for(i=0; i<nQ; i++) k_zvals[i] = INT( (k >> (nQ-1-i)) & 1);  // hack to extract binary(k)
	  for(i=0; i<nQ; i++) zvals[qubits[i]] = k_zvals[i];
	  
	  DEBUG(std::cout << "GETTING CANONICAL AMPLITUDE for B' = " << zvals[0] << " actual=" << comp << std::endl);
	  dcomplex camp = canonical_amplitude_of_target(ip, zvals);
	  DEBUG(std::cout << "GOT CANONICAL AMPLITUDE =" << camp << " updating global amp w/" << comp/camp << std::endl);
	  assert(std::abs(camp) > 1e-6); // Canonical amplitude zero when actual isn't!!
	  _amps[ip] *= comp / camp; // "what we want" / "what stab. frame gives"
	    // this essentially updates a "global phase adjustment factor"
	  break; // move on to next stabilizer state & global amplitude
	}
      }
      if(k == action_size)  assert(false); // Outstate was completely zero!
                                           // (this shouldn't happen if Umx is unitary!)
    }
    DEBUG(std::cout << "Clifford Update END" << std::endl);
  }

  
  dcomplex StateCRep::extract_amplitude(std::vector<INT>& zvals) {
    dcomplex ampl = 0;
    for(INT ip=0; ip < _namps; ip++) {
      ampl += _amps[ip] * canonical_amplitude_of_target(ip, zvals);
    }
    return ampl;
  }

  double StateCRep::measurement_probability(std::vector<INT>& zvals) {
    // Could make this faster in the future by using anticommutator?
    // - maybe could use a _canonical_probability for each ip that is
    //   essentially the 'stabilizer_measurement_prob' fn? -- but need to
    //   preserve *amplitudes* upon measuring & getting output state, which
    //   isn't quite done in the 'pauli_z_meaurement' function.
    dcomplex amp = extract_amplitude(zvals);
    return pow(std::abs(amp),2);
    // Note: don't currently implement the 2nd method using the anticomm in C++... (maybe later)
  }

  void StateCRep::copy_from(StateCRep* other) {
    assert(_n == other->_n && _namps == other->_namps); //make sure we don't need to allocate anything
    INT i;
    for(i=0;i<_2n*_2n;i++) _smatrix[i] = other->_smatrix[i];
    for(i=0;i<_namps*_2n;i++) _pvectors[i] = other->_pvectors[i];
    for(i=0;i<_namps;i++) _amps[i] = other->_amps[i];
    _zblock_start = other->_zblock_start;
    _view_filters.clear();
    for(i=0; i<(INT)other->_view_filters.size(); i++)
      _view_filters.push_back( other->_view_filters[i] );
  }
    

  INT StateCRep::udot1(INT i, INT j) {
    // dot(smatrix[:,i].T, U, smatrix[:,j])
    INT ret = 0;
    for(INT k=0; k < _n; k++)
      ret += _smatrix[(k+_n)*_2n+i] * _smatrix[k*_2n+j];
    return ret;
  }

  void StateCRep::udot2(INT* out, INT* smatrix1, INT* smatrix2) {
    // out = dot(smatrix1.T, U, smatrix2)
    INT tmp;
    for(INT i=0; i<_2n; i++) {
      for(INT j=0; j<_2n; j++) {
	tmp = 0;
	for(INT k=0; k < _n; k++)
	  tmp += smatrix1[(k+_n)*_2n+i] * smatrix2[k*_2n+j];
	out[i*_2n+j] = tmp;
      }
    }
  }
  
  void StateCRep::colsum(INT i, INT j) {
    INT k,row;
    INT* pvec;
    INT* s = _smatrix;
    for(INT p=0; p<_namps; p++) {
      pvec = &_pvectors[ _2n*p ]; // p-th vector
      pvec[i] = (pvec[i] + pvec[j] + 2* udot1(i,j)) % 4;
      for(k=0; k<_n; k++) {
	row = k*_2n;
	s[row+i] = s[row+j] ^ s[row+i];
	row = (k+_n)*_2n;
	s[row+i] = s[row+j] ^ s[row+i];
      }
    }
  }
  
  void StateCRep::colswap(INT i, INT j) {
    INT tmp;
    INT* pvec;
    for(INT k=0; k<_2n; k++) {
      tmp = _smatrix[k*_2n+i];
      _smatrix[k*_2n+i] = _smatrix[k*_2n+j];
      _smatrix[k*_2n+j] = tmp;
    }
    for(INT p=0; p<_namps; p++) {
      pvec = &_pvectors[ _2n*p ]; // p-th vector
      tmp = pvec[i];
      pvec[i] = pvec[j];
      pvec[j] = tmp;
    }
  }
  
  void StateCRep::rref() {
    //Pass1: form X-block (of *columns*)
    INT i=0, j,k,m; // current *column* (to match ref, but our rep is transposed!)
    for(j=0; j<_n; j++) { // current *row*
      for(k=i; k<_n; k++) { // set k = column with X/Y in j-th position
	if(_smatrix[j*_2n+k] == 1) break; // X or Y check
      }
      if(k == _n) continue; // no k found => next column
      colswap(i,k);
      colswap(i+_n,k+_n); // mirror in antistabilizer
      for(m=0; m<_n; m++) {
	if(m != i && _smatrix[j*_2n+m] == 1) { // j-th literal of column m(!=i) is X/Y
	  colsum(m,i);
	  colsum(i+_n,m+_n); // reverse-mirror in antistabilizer (preserves relations)
	}
      }
      i++;
    }
    _zblock_start = i; // first column of Z-block

    //Pass2: form Z-block (of *columns*)
    for(j=0; j<_n; j++) { // current *row*
      for(k=i; k<_n; k++) { // set k = column with Z in j-th position
	if(_smatrix[j*_2n+k] == 0 && _smatrix[(j+_n)*_2n+k] == 1) break; // Z check
      }
      if(k == _n) continue; // no k found => next column
      colswap(i,k);
      colswap(i+_n,k+_n); // mirror in antistabilizer
      for(m=0; m<_n; m++) {
	if(m != i && _smatrix[(j+_n)*_2n+m] == 1) { // j-th literal of column m(!=i) is Z/Y
	  colsum(m,i);
	  colsum(i+_n,m+_n); // reverse-mirror in antistabilizer (preserves relations)
	}
      }
      i++;
    }
  }


  //result = _np.array(zvals_to_acton,INT);
  dcomplex StateCRep::apply_xgen(INT igen, INT pgen, std::vector<INT>& zvals_to_acton,
				   dcomplex ampl, std::vector<INT>& result) {

    dcomplex new_amp = (pgen/2 == 1) ? -ampl : ampl;
    //DEBUG std::cout << "new_amp = "<<new_amp<<std::endl;
    for(std::size_t i=0; i<result.size(); i++)
      result[i] = zvals_to_acton[i];
    
    for(INT j=0; j<_n; j++) { // for each element (literal) in generator
      if(_smatrix[j*_2n+igen] == 1) { // # X or Y
	result[j] = 1-result[j]; //flip!
	// X => a' == a constraint on new/old amplitudes, so nothing to do
	// Y => a' == i*a constraint, so:
	if(_smatrix[(j+_n)*_2n + igen] == 1) { // Y
	  if(result[j] == 1) new_amp *= dcomplex(0,1.0); //+1i; // |0> -> i|1> (but "== 1" b/c result is already flipped)
	  else new_amp *= dcomplex(0,-1.0); //-1i;              // |1> -> -i|0>
	  //DEBUG std::cout << "new_amp2 = "<<new_amp<<std::endl;
	}
      }
      else if(_smatrix[(j+_n)*_2n + igen] == 1) { // Z
	// Z => a' == -a constraint if basis[j] == |1> (otherwise a == a)
	if(result[j] == 1) new_amp *= -1.0;
	//DEBUG std::cout << "new_amp3 = "<<new_amp<<std::endl;
      }
    }
    //DEBUG std::cout << "new_amp4 = "<<new_amp<<std::endl;
    return new_amp;
  }
        
  dcomplex StateCRep::get_target_ampl(std::vector<INT>& tgt, std::vector<INT>& anchor, dcomplex anchor_amp, INT ip) {
    // requires just a single pass through X-block
    std::vector<INT> zvals(anchor);
    dcomplex amp = anchor_amp; //start with anchor state
    INT i,j,k, lead = -1;
    DEBUG(std::cout << "BEGIN get_target_ampl" << std::endl);
    
    for(i=0; i<_zblock_start; i++) { // index of current generator
      INT gen_p = _pvectors[ip*_2n + i]; //phase of generator
      gen_p = (gen_p + 3*udot1(i,i)) % 4;  //counts number of Y's => -i's
      assert(gen_p == 0 || gen_p == 2); //Logic error: phase should be +/- only!
      
      // Get leading flipped qubit (lowest # qubit which will flip when we apply this)
      for(j=0; j<_n; j++) {
	if(_smatrix[j*_2n+i] == 1) { // check for X/Y literal in qubit pos j
	  assert(j > lead); // lead should be strictly increasing as we iterate due to rref structure
	  lead = j; break;
	}
      }
      if(j == _n) assert(false); //Should always break loop!

      DEBUG(std::cout << "GETTGT iter " << i << " lead=" << lead << " genp=" << gen_p << " amp=" << amp << std::endl);

      //Check whether we should apply this generator to zvals
      if(zvals[lead] != tgt[lead]) {
	// then applying this generator is productive - do it!
	DEBUG(std::cout << "Applying XGEN amp=" << amp << std::endl);
	std::vector<INT> zvals_copy(zvals);
	amp = apply_xgen(i, gen_p, zvals, amp, zvals_copy);
	zvals = zvals_copy; //will this work (copy)?

	//DEBUG!!! - print XGEN return val
	//std::cout << "Resulting amp = " << amp << " zvals=";
        //for(std::size_t z=0; z<zvals.size(); z++) std::cout << zvals[z];
	//std::cout << std::endl;
                    
	// Check if we've found target
	for(k=0; k<_n; k++) {
	  if(zvals[k] != tgt[k]) break;
	}
	if(k == _n) {
	  DEBUG(std::cout << "FOUND!" << std::endl);
	  return amp; // no break => (zvals == tgt)
	}
      }
    }
    assert(false); //Falied to find amplitude of target! (tgt)
    return 0; // just to avoid warning
  }
  
  dcomplex StateCRep::canonical_amplitude_of_target(INT ip, std::vector<INT>& target) {
    rref(); // ensure we're in reduced row echelon form
        
    // Stage1: go through Z-block columns and find an "anchor" - the first
    // basis state that is allowed given the Z-block parity constraints.
    // (In Z-block, cols can have only Z,I literals)
    INT i,j;
    DEBUG(std::cout << "CanonicalAmps STAGE1: zblock_start = " << _zblock_start << std::endl);
    std::vector<INT> anchor(_n); // "anchor" basis state (zvals), which gets amplitude 1.0 by definition
    for(i=0; i<_n; i++) anchor[i] = 0;
    
    INT lead = _n;
    for(i=_n-1; i >= _zblock_start; i--) { //index of current generator
      INT gen_p = _pvectors[ip*_2n + i]; //phase of generator
      gen_p = (gen_p + 3*udot1(i,i)) % 4;  //counts number of Y's => -i's
      assert(gen_p == 0 || gen_p == 2);
      DEBUG(std::cout << "STARTING LOOP!" << std::endl);
            
      // get positions of Zs
      std::vector<INT> zpos;
      for(j=0; j<_n; j++) {
	if(_smatrix[(j+_n)*_2n+i] == 1) zpos.push_back(j);
      }      

      // set values of anchor between zpos[0] and lead
      // (between current leading-Z position and the last iteration's,
      //  which marks the point at which anchor has been initialized to)
      INT fixed1s = 0; // relevant number of 1s fixed by the already-initialized part of 'anchor'
      INT target1s = 0; // number of 1s in target state, which we want to check for Z-block compatibility
      std::vector<INT> zpos_to_fill;
      std::vector<INT>::iterator it;
      for(it=zpos.begin(); it!=zpos.end(); ++it) {
	j = *it;
	if(j >= lead) {
	  if(anchor[j] == 1) fixed1s += 1;
	}
	else zpos_to_fill.push_back(j);

	if(target[j] == 1) target1s += 1;
      }

      assert(zpos_to_fill.size() > 0); // structure of rref Z-block should ensure this
      INT parity = gen_p/2;
      INT eff_parity = (parity - (fixed1s % 2)) % 2; // effective parity for zpos_to_fill

      DEBUG(std::cout << "  Current gen = "<<i<<" phase = "<<gen_p<<" zpos="<<zpos.size()<<" fixed1s="<<fixed1s<<" tofill="<<zpos_to_fill.size()<<" eff_parity="<<eff_parity<<" lead="<<lead << std::endl);
      DEBUG(std::cout << "   -anchor: ");
      DEBUG(for(INT dbi=0; dbi<_n; dbi++) std::cout << anchor[dbi] << "  ");
      
      if((target1s % 2) != parity)
	return dcomplex(0.0); // target fails this parity check -> it's amplitude == 0 (OK)
      
      if(eff_parity == 0) { // even parity - fill with all 0s
	// BUT already initalized to 0s, so don't need to do anything for anchor
      }
      else { // odd parity (= 1 or -1) - fill with all 0s except final zpos_to_fill = 1
	anchor[zpos_to_fill[zpos_to_fill.size()-1]] = 1; // BUT just need to fill in the final 1
      }
      lead = zpos_to_fill[0]; // update the leading-Z index
      DEBUG(std::cout << "   ==> ");
      DEBUG(for(INT dbi=0; dbi<_n; dbi++) std::cout << anchor[dbi] << "  ");
      DEBUG(std::cout << std::endl);
    }
    
    //Set anchor amplitude to appropriate 1.0/sqrt(2)^s 
    // (by definition - serves as a reference pt)
    // Note: 's' equals the minimum number of generators that are *different*
    // between this state and the basis state we're extracting and ampl for.
    // Since any/all comp. basis state generators can form all and only the
    // Z-literal only (Z-block) generators 's' is simplly the number of
    // X-block generators (= self.zblock_start).
    INT s = _zblock_start;
    dcomplex anchor_amp = 1/(pow(sqrt(2.0),s));
    
    //STAGE 2b - for sampling a set
    // Check exit conditions
    DEBUG(std::cout << "CanonicalAmps STAGE2" << std::endl);
    for(i=0; i<_n; i++) {
      if(anchor[i] != target[i]) break;
    }
    if(i == _n) return anchor_amp; // no break => (anchor == target)
    
    // Stage2: move through X-block processing existing amplitudes
    // (or processing only to move toward a target state?)
    DEBUG(std::cout << "Getting target ampl" << std::endl);
    return get_target_ampl(target,anchor,anchor_amp,ip);
  }
    
  void StateCRep::canonical_amplitudes_sample(INT ip, std::vector<INT> qs_to_sample,
						std::vector<INT>& anchor, std::vector<dcomplex>& amp_samples) {
    rref(); // ensure we're in reduced row echelon form

    INT i,j,k;
    INT remaining = pow(2,qs_to_sample.size()); //number we still need to find
    assert(amp_samples.size() == remaining);
    for(i=0; i<remaining; i++) amp_samples[i]= std::nan("empty slot");
    // what we'll eventually return - holds amplitudes of all
    //  variations of qs_to_sample starting from anchor.
        
    // Stage1: go through Z-block columns and find an "anchor" - the first
    // basis state that is allowed given the Z-block parity constraints.
    // (In Z-block, cols can have only Z,I literals)
    assert(anchor.size() == _n); // "anchor" basis state (zvals), which gets amplitude 1.0 by definition
    for(i=0; i<_n; i++) anchor[i] = 0;
    
    INT lead = _n;
    for(i=_n-1; i >= _zblock_start; i--) { //index of current generator
      INT gen_p = _pvectors[ip*_2n + i]; //phase of generator
      gen_p = (gen_p + 3*udot1(i,i)) % 4;  //counts number of Y's => -i's
      assert(gen_p == 0 || gen_p == 2);
            
      // get positions of Zs
      std::vector<INT> zpos;
      for(j=0; j<_n; j++) {
	if(_smatrix[(j+_n)*_2n+i] == 1) zpos.push_back(j);
      }

      // set values of anchor between zpos[0] and lead
      // (between current leading-Z position and the last iteration's,
      //  which marks the point at which anchor has been initialized to)
      INT fixed1s = 0; // relevant number of 1s fixed by the already-initialized part of 'anchor'
      std::vector<INT> zpos_to_fill;
      std::vector<INT>::iterator it;
      for(it=zpos.begin(); it!=zpos.end(); ++it) {
	j = *it;
	if(j >= lead) {
	  if(anchor[j] == 1) fixed1s += 1;
	}
	else zpos_to_fill.push_back(j);
      }
	
      assert(zpos_to_fill.size() > 0); // structure of rref Z-block should ensure this
      INT parity = gen_p/2;
      INT eff_parity = (parity - (fixed1s % 2)) % 2; // effective parity for zpos_to_fill

      if(eff_parity == 0) { // even parity - fill with all 0s
	// BUT already initalized to 0s, so don't need to do anything for anchor
      }
      else { // odd parity (= 1 or -1) - fill with all 0s except final zpos_to_fill = 1
	anchor[zpos_to_fill[zpos_to_fill.size()-1]] = 1; // BUT just need to fill in the final 1
      }
      lead = zpos_to_fill[0]; // update the leading-Z index
    }
    
    //Set anchor amplitude to appropriate 1.0/sqrt(2)^s 
    // (by definition - serves as a reference pt)
    // Note: 's' equals the minimum number of generators that are *different*
    // between this state and the basis state we're extracting and ampl for.
    // Since any/all comp. basis state generators can form all and only the
    // Z-literal only (Z-block) generators 's' is simplly the number of
    // X-block generators (= self.zblock_start).
    INT s = _zblock_start;
    dcomplex anchor_amp = 1/(pow(sqrt(2.0),s));
    
    remaining -= 1;
    INT nk = qs_to_sample.size();
    INT anchor_indx = 0;
    for(k=0; k<nk; k++) anchor_indx += anchor[qs_to_sample[k]]*pow(2,(nk-1-k));
    amp_samples[ anchor_indx ] = anchor_amp;
    
    
    //STAGE 2b - for sampling a set
    
    //If we're trying to sample a set, check if any of the amplitudes
    // we're looking for are zero by the Z-block checks.  That is,
    // consider whether anchor with qs_to_sample indices updated
    // passes or fails each check
    for(i=_n-1; i >= _zblock_start; i--) { // index of current generator
      INT gen_p = _pvectors[ip*_2n + i]; //phase of generator
      gen_p = (gen_p + 3*udot1(i,i)) % 4;  //counts number of Y's => -i's
      
      std::vector<INT> zpos;
      for(j=0; j<_n; j++) {
	if(_smatrix[(j+_n)*_2n+i] == 1) zpos.push_back(j);
      }
      
      std::vector<INT> inds;
      std::vector<INT>::iterator it, it2;
      INT fixed1s = 0; // number of 1s in target state, which we want to check for Z-block compatibility
      for(it=zpos.begin(); it!=zpos.end(); ++it) {
	j = *it;
	it2 = std::find(qs_to_sample.begin(),qs_to_sample.end(),j);
	if(it2 != qs_to_sample.end()) { // if j in qs_to_sample
	  INT jpos = it2 - qs_to_sample.begin();
	  inds.push_back( jpos ); // "sample" indices in parity check
	}
	else if(anchor[j] == 1) {
	  fixed1s += 1;
	}
      }
      if(inds.size() > 0) {
	INT parity = (gen_p/2 - (fixed1s % 2)) % 2; // effective parity
	INT* b = new INT[qs_to_sample.size()]; //els are just 0 or 1
	INT bi;
	for(bi=0; bi<(INT)qs_to_sample.size(); bi++) b[bi] = 0;
	k = 0;
	while(true) {
	  // tup == b
	  INT tup_parity = 0;
	  for(INT kk=0; kk<(INT)inds.size(); kk++) tup_parity += b[inds[kk]];
	  if(tup_parity != parity) { // parity among inds is NOT allowed => set ampl to zero
	    if(std::isnan(amp_samples[k].real())) remaining -= 1; //need NAN check here -- TODO replace -1 sentinels
	    amp_samples[k] = 0.0;
	  }
	  
	  k++; // increment k
	  
	  // increment b ~ itertools.product
	  for(bi=qs_to_sample.size()-1; bi >= 0; bi--) {
	    if(b[bi]+1 < 2) { // 2 == number of indices, i.e. [0,1]
	      b[bi] += 1;
	      break;
	    }
	    else {
	      b[bi] = 0;
	    }
	  }
	  if(bi < 0) break;  // if didn't break out of loop above, then can't
	}                    // increment anything - break while(true) loop.
	delete [] b;
      }
    }
    
    // Check exit conditions
    if(remaining == 0) return;
    
    // Stage2: move through X-block processing existing amplitudes
    // (or processing only to move toward a target state?)

    std::vector<INT> target(anchor);
    INT* b = new INT[qs_to_sample.size()]; //els are just 0 or 1
    INT bi;
    for(bi=0; bi<(INT)qs_to_sample.size(); bi++) b[bi] = 0;
    k = 0;
    while(true) {
      // tup == b
      if(std::isnan(amp_samples[k].real())) {
	for(INT kk=0; kk<(INT)qs_to_sample.size(); kk++)
	  target[qs_to_sample[kk]] = b[kk];
	amp_samples[k] = get_target_ampl(target,anchor,anchor_amp,ip);
      }
      
      k++; // increment k
      
      // increment b ~ itertools.product
      for(bi=qs_to_sample.size()-1; bi >= 0; bi--) {
	if(b[bi]+1 < 2) { // 2 == number of indices, i.e. [0,1]
	  b[bi] += 1;
	  break;
	}
	else {
	  b[bi] = 0;
	}
      }
      if(bi < 0) break;  // if didn't break out of loop above, then can't
    }
    delete [] b;
    return;
  }

  void StateCRep::apply_clifford_to_frame(INT* s, INT* p, std::vector<INT> qubit_filter) {
    //for now, just embed s,p inside full-size s,p: (TODO: make this function more efficient!)
    INT* full_s = new INT[_2n*_2n];
    INT* full_p = new INT[_2n];

    // Embed s,p inside full_s and full_p
    INT i,j,ne = qubit_filter.size();
    INT two_ne = 2*ne;
    for(i=0; i<_2n; i++) {
      for(j=0; j<_2n; j++) full_s[i*_2n+j] = (i==j) ? 1 : 0; // full_s == identity
    }
    for(i=0; i<_2n; i++) full_p[i] = 0; // full_p = zero
    
    for(INT ii=0; ii<ne; ii++) {
      i = qubit_filter[ii];
      full_p[i] = p[ii];
      full_p[i+_n] = p[ii+ne];
      
      for(INT jj=0; jj<ne; jj++) {
	j = qubit_filter[jj];
	full_s[i*_2n+j] = s[ii*two_ne+jj];
	full_s[(i+_n)*_2n+j] = s[(ii+ne)*two_ne+jj];
	full_s[i*_2n+(j+_n)] = s[ii*two_ne+(jj+ne)];
	full_s[(i+_n)*_2n+(j+_n)] = s[(ii+ne)*two_ne+(jj+ne)];
      }
    }

    apply_clifford_to_frame(full_s, full_p);

    delete [] full_s;
    delete [] full_p;
  }
    
  
  void StateCRep::apply_clifford_to_frame(INT* s, INT* p) {
    INT i,j,k,tmp;
    
    // Below we calculate the s and p for the output state using the formulas from
    // Hostens and De Moor PRA 71, 042315 (2005).

    // out_s = _mtx.dotmod2(s,self.s)
    INT* out_s = new INT[_2n*_2n];
    //if(qubit_filter.size() == 0) {
    for(i=0; i<_2n; i++) {
      for(j=0; j<_2n; j++) {
	tmp = 0;
	for(k=0; k<_2n; k++) // row(s, i) * col(_smatrix,j)
	  tmp += s[i*_2n+k] * _smatrix[k*_2n+j];
	out_s[i*_2n+j] = tmp % 2; // all els are mod(2)
      }
    }
    //} else {
    //  INT ii;
    //  INT ne = qubit_filter.size(); // number of qubits s,p act on
    //  
    //  //use qubit_filter - only rows & cols of "full s" corresponding to qubit_filter are non-identity
    //  for(i=0; i<_2n*_2n; i++) out_s[i] = _smatrix[i]; // copy out_s = _smatrix
    //
    //  for(ii=0; ii<qubit_filter.size(); ii++) { // only non-identity rows of "full s"
    //	i = qubit_filter[ii];
    //    for(j=0; j<_2n; j++) {
    //	  tmp = 0;
    //	  for(INT kk=0; kk<qubit_filter.size(); kk++) { // only non-zero cols of non-identity i-th row of "full s"
    //	    k = qubit_filter[kk];
    //	    tmp += s[ii*_2n+kk] * _smatrix[k*_2n+j];
    //	    tmp += s[ii*_2n+(kk+ne)] * _smatrix[(k+_n)*_2n+j];
    //	  }
    //	  out_s[i*_2n+j] = tmp % 2; // all els are mod(2)
    //    }
    //
    //	// part2, for (i+n)-th row of "full s"
    //	i = qubit_filter[ii] + _n;
    //	INT iin = ii + ne;
    //	for(j=0; j<_2n; j++) {
    //	  tmp = 0;
    //	  for(INT kk=0; kk<qubit_filter.size(); kk++) { // only non-zero cols of non-identity i-th row of "full s"
    //	    k = qubit_filter[kk];
    //	    tmp += s[iin*_2n+kk] * _smatrix[k*_2n+j];
    //	    tmp += s[iin*_2n+(kk+ne)] * _smatrix[(k+_n)*_2n+j];
    //	  }
    //	  out_s[i*_2n+j] = tmp % 2; // all els are mod(2)
    //    }
    //  }
    //}

    INT* inner = new INT[_2n*_2n];
    INT* tmp1 = new INT[_2n];
    INT* tmp2 = new INT[_2n*_2n];
    INT* vec = new INT[_2n];
    udot2(inner, s, s);

    // vec = _np.dot(_np.transpose(_smatrix),p - _mtx.diagonal_as_vec(inner))
    for(i=0; i<_2n; i++) tmp1[i] = p[i] - inner[i*_2n+i];
    for(i=0; i<_2n; i++) {
      vec[i] = 0; 
      for(k=0; k<_2n; k++)
	vec[i] += _smatrix[k*_2n+i] * tmp1[k];
    }
	  
    //matrix = 2*_mtx.strictly_upper_triangle(inner)+_mtx.diagonal_as_matrix(inner)
    INT* matrix = inner; //just modify inner in place since we don't need it anymore
    for(i=0; i<_2n; i++) {
      for(j=0; j<i; j++) matrix[i*_2n+j] = 0; //lower triangle
      for(j=i+1; j<_2n; j++) matrix[i*_2n+j] *= 2; //strict upper triangle
    }

    //vec += _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(self.s),matrix),self.s))
    for(i=0; i<_2n; i++) {
      for(j=0; j<_2n; j++) {
	tmp2[i*_2n+j] = 0;
	for(k=0; k<_2n; k++)
	  tmp2[i*_2n+j] += _smatrix[k*_2n+i]*matrix[k*_2n+j];
      }
    }
    for(i=0; i<_2n; i++) {  //TODO - could put this within i-loop above and only use tmp1...
      for(k=0; k<_2n; k++)
	vec[i] += tmp2[i*_2n+k]*_smatrix[k*_2n+i];
    }

    // _smatrix = out_s (don't set this until we're done using _smatrix)
    for(i=0; i<_2n*_2n; i++) _smatrix[i] = out_s[i];
    for(i=0; i<_namps; i++) {
      INT* pvec = &_pvectors[ _2n*i ]; // i-th vector
      for(k=0; k<_2n; k++) pvec[k] = (pvec[k] + vec[k]) % 4;
    }

    delete [] out_s;
    delete [] inner;
    delete [] tmp1;
    delete [] tmp2;
    delete [] vec;
  }

  void StateCRep::print(const char* label) {
    std::cout << "<" << label << " (StateCRep - TODO print)>" << std::endl;
  }

}
