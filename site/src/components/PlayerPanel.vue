<template>
    <table class="panel" >
        <tr>
            <td class="side">
                <img v-if="side==0" class="stone" src="../assets/stoneB.svg">
                <img v-if="side==1" class="stone" src="../assets/stoneW.svg">
            </td>
            <td :class="{ 'clock': true, 'running': running, }">
                {{minutes}}
            </td>
        </tr>
        <tr>
            <td class="eval" colspan="2">{{eval}}</td>
        </tr>
        <tr v-if="depth != null">
            <td class="eval" colspan="2">depth: {{depth}}</td>
        </tr>
        <tr v-if="nodes != null">
            <td class="eval" colspan="2">nodes: {{nodes}}</td>
        </tr>
    </table>
</template>

<script>
    import store from "../store"

    export default {
        props: ['side'],
        data () {
            return {
                gameStatus: store.gameStatus,
                time: 0,
                prevTick: 0,
                running: false,
                value: null,
                depth: null,
                nodes: null,
                timerId: null
                }
            },
        computed: {
            minutes: function(){
                let totalSeconds = Math.floor(this.time / 1000);
                let minutes = Math.floor(totalSeconds / 60)
                let seconds = totalSeconds % 60;
                if(seconds < 10){
                    return minutes + ':0' + seconds;
                } else{
                    return minutes + ':' + seconds;
                }
            },
            eval: function(){
                if (this.value){
                    return "eval: " + (Math.round(this.value * 1000) / 1000)
                } else {
                    return "";
                }
            }
        },
        watch: {
            'gameStatus.lastMove'(newLastMove, oldLastMove) {
                if(!newLastMove){
                    this.time = 0;
                    this.value = null;
                    this.depth = null;
                }
            },
            'gameStatus.sideToMove'(newSideToMove, oldSideToMove){
                if(newSideToMove == this.side && oldSideToMove != this.side && this.gameStatus.inProgress){
                    this.running = true;
                    this.prevTick = Date.now()
                    let self = this;
                    this.timerId = setInterval(function() {
                        let now = Date.now(); 
                        let elapsed = now - self.$data.prevTick;
                        self.$data.time += elapsed;
                        self.$data.prevTick = now; 
                    }, 200);
                }
                if(newSideToMove != this.side && oldSideToMove == this.side){
                    this.running = false;
                    if(this.timerId){
                        clearInterval(this.timerId);
                    }
                }
                if(newSideToMove != this.side){
                    this.value = this.gameStatus.value;
                    this.depth = this.gameStatus.depth;
                    this.nodes = this.gameStatus.nodes;
                }
            }
        }
}
</script>

<style scoped>
    .panel {
        border-spacing: 0;
        border-collapse: collapse;
        width: 100px;
    }
    .running{
        border: 3px solid rgb(0, 0, 0)!important;
    }
    .side {
        margin: 0;
        padding: 5px; 
        width: 40px;
        height: 40px;
        border: 1px solid rgb(0, 0, 0);
    }
    .clock{
        text-align: center;
        border: 1px solid rgb(0, 0, 0);
    }
    .stone{
        display: block;
        width: 100%;
        height: 100%;
    }
</style>