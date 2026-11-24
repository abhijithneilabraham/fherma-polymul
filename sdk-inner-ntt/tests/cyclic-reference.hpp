#pragma once
static void cyclic(std::vector<U>& a,U root,U p) {
    const unsigned n=a.size();
    for(unsigned i=1,j=0;i<n;++i) {
        unsigned bit=n/2;
        while(j&bit) {j^=bit;bit/=2;}
        j^=bit;
        if(i<j) std::swap(a[i],a[j]);
    }
    for(unsigned width=2;width<=n;width*=2) {
        U step=pow_mod(root,n/width,p);
        for(unsigned begin=0;begin<n;begin+=width) {
            U factor=1;
            for(unsigned j=0;j<width/2;++j) {
                U x=a[begin+j],y=product(a[begin+j+width/2],factor,p);
                a[begin+j]=(uint64_t(x)+y)%p;
                a[begin+j+width/2]=(uint64_t(x)+p-y)%p;
                factor=product(factor,step,p);
            }
        }
    }
}

