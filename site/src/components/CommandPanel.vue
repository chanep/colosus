<template>
  <div class="container">
    <div class="settings">
      <div class="time-control">
        <div class="time-control-item">
          <input name="mode-iteration" true-value="mode-iterations" false-value="mode-time" v-model="mode" type="checkbox" value="mode-iterations">
          <div>
            <label for="iterations">Iterations</label>
            <input name="iterations" v-model.number="iterations" type="number" :disabled="mode == 'mode-iterations' ? false : true">
          </div>
        </div>
        <div class="time-control-item">
          <input name="mode-time" true-value="mode-time" false-value="mode-iterations" v-model="mode" type="checkbox" value="mode-time">
          <div>
            <label for="time">Time</label>
            <input name="time" v-model.number="time" type="number" :disabled="mode == 'mode-time' ? false : true">
          </div>
        </div>
      </div>

      <div class="side-control">
        <div class="radio-group">
          <input type="radio" v-model="humanSide" :value="humanSideValues.black">
          <label>Black</label>
        </div>
        <div class="radio-group">
          <input type="radio" v-model="humanSide" :value="humanSideValues.white">
          <label>White</label>
        </div>
        <div class="radio-group">
          <input type="radio" v-model="humanSide" :value="humanSideValues.both">
          <label>Human vs. Human</label>
        </div>
        <div class="radio-group">
          <input type="radio" v-model="humanSide" :value="humanSideValues.none">
          <label>Colosus vs. Colosus</label>
        </div>
      </div>
    </div>

    <div>
      <button @click="newGame">New Game</button>
    </div>
  </div>
</template>

<script>
import store from "../store";
import api from "../api";

export default {
  data() {
    return {
      iterations: 256,
      time: 1,
      mode: "mode-iterations",
      humanSide: "black",
      humanSideValues: {
        black: "black",
        white: "white",
        both: "both",
        none: "none"
      }
    };
  },
  methods: {
    newGame() {
      let blackHuman = this.humanSide == "black" || this.humanSide == "both";
      let whiteHuman = this.humanSide == "white" || this.humanSide == "both";
      let iterations = this.iterations;
      if(this.mode == "mode-time")
        iterations = 0;

      api.newGame(blackHuman, whiteHuman, iterations, this.time);
    }
  }
};
</script>

<style scoped>
.container {
  display: flex;
  flex-direction: column;
  padding: 20px;
}
.settings {
  display: flex;
  flex-direction: row;
  width: 300px;
  height: 120px;
  /* border: 1px solid black; */
  margin-bottom: 30px;
}
.time-control {
  display: flex;
  flex-direction: column;
  flex-basis: 30%;
  justify-content: space-between;
  padding-right: 10px
}

.time-control-item {
  display: flex;
  flex-direction: row;
  align-items: flex-end;
  padding-bottom: 10px;
}

.time-control-item > input[type="checkbox"]{
  width: 30px;
}
.time-control-item input[type="number"]{
  width: 80px;
}
.side-control {
  display: flex;
  flex-direction: column;
  flex-basis: 70%;
  justify-content: space-between;
}
.text-group {
  display: flex;
  flex-direction: column;
  margin-bottom: 10px;
}
.radio-group {
  display: flex;
  flex-direction: row;
}
input[type="radio"] {
  margin-right: 5px;
}
</style>