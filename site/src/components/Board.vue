<template>
        <table class="board">
            <tr v-for="(rank, rindex) in reversedBoard">
                <td class="square" v-for="(square, findex) in rank" @click="move(14 - rindex, findex)">
                    <img v-if="square=='X'" class="stone" src="../assets/stoneB.svg">
                    <img v-if="square=='O'" class="stone" src="../assets/stoneW.svg">
                </td>
                <td class="coord_r">{{15 - rindex}}</td>
            </tr>
            <tr>
                <td class="coord_f" v-for="file in 15">{{file}}</td>
            </tr>
        </table>
</template>

<script>
    import store from "../store"
    import api from "../api"

    export default {
        data () {
            return {
                gameStatus: store.gameStatus,
                ticker: "x"
                }
            },
        computed: {
            reversedBoard: function(){
                return this.gameStatus.board.reverse()
            }
        },
        methods:{
            move(rank, file){
                var self = this;

                // setInterval(function() {
                //     self.$data.ticker = Date.now(); 
                //     console.log('tickerrrr');
                // }, 1000);

                api.move(rank, file);
            }
        }


}
</script>

<style scoped>
    .board {
        border-spacing: 0;
        border-collapse: collapse;
        
    }
    .square {
        margin: 0;
        padding: 2px; 
        border: 1px solid rgb(0, 0, 0);
        width: 20px;
        height: 20px;
        background-color: rgb(255, 223, 193);
    }

    .coord_r {
        padding: 1px; 
        width: 20px;
        height: 20px;
    }
    .coord_f {
        text-align: center;
        padding: 1px; 
        width: 20px;
        height: 20px;
    }
    .stone{
        display: block;
        width: 100%;
        height: 100%;
    }
</style>


