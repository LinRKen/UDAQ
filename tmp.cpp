#include<bits/stdc++.h>
using namespace std;

class Solution {
public:
    static const int MAXN = 1e5+10;
    int f[MAXN][3][3];
    int time[MAXN];

    const bool cmp(const vector<int> &A,const vector<int> &B){
    return A[0]<B[0];
    }
    
    int getMaximumNumber(vector<vector<int>>& arr) {
    sort(arr.begin(),arr.end(),cmp);
    memset(f,-1,sizeof(f));
    f[0][1][1] =0;
    int last=0
    for (int i=0;i<n;i++)
        {
            for (int a1=0;a1<3;a1++)
                for (int b1=0;b1<3;b1++)
                if ( f[i][a1][b1] != -1 )
                    for (int a2=0;a2<3;a2++)
                        for (int b2=0;b2<3;b2++)
                            if ( abs(a1-a2)+abs(b1-b2) <= arr[i][0]-last)
                            {
                                int val = ( (arr[i][1] == a2 ) && ( arr[i][2] == b2 ) );
                                f[i+1][a2][b2] = max(f[i+1][a2][b2] , f[i][a1][b1]+ val );
                            }
        }
    int ans =0;
    for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
            ans = max(ans, f[n][i][j]);
    return ans;
    }
};